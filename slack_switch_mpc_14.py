import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
A = 1.0
S_out = 0.5
g = 9.81

dt = 0.5
T = 60
t_grid = np.arange(0, T, dt)

nx = 1
nu = 1

h_min, h_max = 0.0, 0.8
u_min, u_max = 0.0, 1.5

N_long = 15
N_short = 5

s_high = 0.08
s_low = 0.02
alpha = 0.9
min_dwell = 5

# =========================
# 変更点①：平滑化モデル
# =========================
def smooth_relu(x, eps=1e-4):
    return 0.5 * (x + ca.sqrt(x**2 + eps))

def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h = smooth_relu(x[0])
    qi = smooth_relu(u[0] + w)

    h_dot = -(S_out/A)*ca.sqrt(2*g*h + 1e-6) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

# =========================
# 変更点②：RK4
# =========================
def rk4(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def rk4_sym(x, u):
    k1 = f(x, u, 0)
    k2 = f(x + dt/2*k1, u, 0)
    k3 = f(x + dt/2*k2, u, 0)
    k4 = f(x + dt*k3, u, 0)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# =========================
# MPC builder
# =========================
def build_mpc(N):
    X = ca.SX.sym("X", nx, N+1)
    U = ca.SX.sym("U", nu, N)

    J = 0
    G = []

    for k in range(N):
        J += 10 * ca.sumsqr(X[k] - 0.8)
        J += 0.1 * ca.sumsqr(U[k])

        # 変更点③：Δu
        if k > 0:
            J += 0.5 * 10.0 * ca.sumsqr(U[k] - U[k-1])

        G.append(X[k+1] - rk4_sym(X[k], U[k]))

    J += 10 * ca.sumsqr(X[N] - 0.8)

    opt = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)))
    return ca.nlpsol("solver", "ipopt",
                     {"x": opt, "f": J, "g": ca.vertcat(*G)},
                     {"ipopt.print_level": 0, "print_time": 0})

solver_long = build_mpc(N_long)
solver_short = build_mpc(N_short)

# =========================
def disturbance(t):
    return -0.8 if 20 <= t <= 25 else 0.0

# =========================
# ★ x0 長さ保証（最重要）
# =========================
def adjust_x0_length(x0, target_len):
    x0 = np.asarray(x0).ravel()
    if x0.size > target_len:
        return x0[:target_len]
    elif x0.size < target_len:
        pad = np.full(target_len - x0.size, x0[-1])
        return np.concatenate([x0, pad])
    return x0

# =========================
def simulate(variable=True):
    x = ca.DM([0.8])
    x_pred_prev = x

    s_bar = 0.0
    mode = "long"
    dwell = 0

    x0 = None
    prev_solx = None

    X_log, U_log, N_log, S_log = [], [], [], []

    for k, t in enumerate(t_grid):
        w = disturbance(t)

        s = abs(float(x - x_pred_prev))
        s_bar = alpha*s_bar + (1-alpha)*s

        if variable and dwell == 0:
            if mode != "short" and s_bar > s_high:
                mode, dwell = "short", min_dwell
            elif mode != "long" and s_bar < s_low:
                mode, dwell = "long", min_dwell

        dwell = max(dwell-1, 0)

        if not variable:
            mode = "long"

        solver, N = (solver_short, N_short) if mode == "short" else (solver_long, N_long)
        nxu = nx*(N+1) + nu*N

        if prev_solx is None:
            x0 = [float(x)] * nxu
        else:
            # 変更点⑤：先生指定の形
            x0 = np.concatenate([prev_solx[nx:], prev_solx[-nu:]])

        # ★ 必ず長さを合わせる
        x0 = adjust_x0_length(x0, nxu)

        lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
        ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
        lbx[0] = ubx[0] = float(x)

        sol = solver(
            lbx=lbx, ubx=ubx,
            lbg=[0]*(nx*N), ubg=[0]*(nx*N),
            x0=x0
        )

        solx = sol["x"].full().ravel()
        prev_solx = solx

        u = ca.DM([solx[nx*(N+1)]])
        x_pred_prev = ca.DM([solx[nx]])

        x = rk4(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(s_bar)

    return np.array(X_log), np.array(U_log), np.array(N_log), np.array(S_log)

# =========================
Xf, Uf, Nf, Sf = simulate(variable=False)
Xv, Uv, Nv, Sv = simulate(variable=True)

# =========================
plt.figure(figsize=(8,8))

plt.subplot(4,1,1)
plt.plot(t_grid, Xf, "b--", label="Fixed")
plt.plot(t_grid, Xv, "r", label="Variable")
plt.ylabel("h")
plt.legend()

plt.subplot(4,1,2)
plt.plot(t_grid, Uf, "b--")
plt.plot(t_grid, Uv, "r")
plt.ylabel("u")

plt.subplot(4,1,3)
plt.plot(t_grid, Nv)
plt.ylabel("N")

plt.subplot(4,1,4)
plt.plot(t_grid, Sv)
plt.ylabel("Slack (EMA)")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()

