import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# ==============================
# 基本設定
# ==============================
A = 1.0
S_out = 0.1
g = 9.81

nx, nu = 1, 1
dt = 0.2
t_end = 100.0
t_grid = np.arange(0, t_end, dt)

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])

# ★① ホライズン短縮
N_long  = 8
N_short = 3

# Δu ペナルティ
Rdu_fixed    = 1.0
Rdu_variable = 10.0

# ヒステリシス
s_high = 5e-3
s_low  = 1e-4
min_dwell = 5

# ★③ 制約を厳しく
h_min, h_max = 0.2, 1.2
u_min, u_max = 0.0, 1.2

# ==============================
# ★② 外乱（複数回）
# ==============================
def disturbance(t):
    if 20 <= t <= 25:
        return -1.0
    if 40 <= t <= 45:
        return -0.8
    if 70 <= t <= 80:
        return -1.2
    return 0.0

# ==============================
# 平滑化モデル
# ==============================
def smooth_relu(x, eps=1e-4):
    return 0.5 * (x + ca.sqrt(x**2 + eps))

def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h  = smooth_relu(x[0])
    qi = smooth_relu(u[0] + w)

    h_dot = -(S_out/A) * ca.sqrt(2*g*h + 1e-6) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

# ==============================
# RK4
# ==============================
def rk4(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return x + dt*(k1+2*k2+2*k3+k4)/6

def rk4_sym(x, u):
    return rk4(x, u, 0)

# ==============================
# ★④ MPCソルバ（Δu重みを分離）
# ==============================
def make_solver(N, Rdu):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]

    J, G = 0, []

    for k in range(N):
        J += 0.5 * ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])
        J += 0.5 * ca.mtimes([U[k].T, R, U[k]])
        if k > 0:
            J += 0.5 * Rdu * ca.sumsqr(U[k] - U[k-1])
        G.append(X[k+1] - rk4_sym(X[k], U[k]))

    J += 0.5 * ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])

    return ca.nlpsol("solver", "ipopt",
        {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)},
        {"ipopt.print_level": 0, "print_time": 0}
    )

solver_fixed = make_solver(N_long, Rdu_fixed)
solver_long  = make_solver(N_long, Rdu_variable)
solver_short = make_solver(N_short, Rdu_variable)

# ==============================
# シミュレーション
# ==============================
def simulate(variable=True):
    x = ca.DM([0.8])
    x_pred_prev = x

    mode, dwell, s_bar = "long", 0, 0.0
    X_log, U_log, N_log, S_log, T_log = [], [], [], [], []
    x0 = None

    for t in t_grid:
        w = disturbance(t)

        s = abs(float(x - x_pred_prev))
        s_bar = 0.9*s_bar + 0.1*s

        if variable and dwell == 0:
            if mode == "long" and s_bar > s_high:
                mode, dwell = "short", min_dwell
            elif mode == "short" and s_bar < s_low:
                mode, dwell = "long", min_dwell

        dwell = max(dwell-1, 0)

        if not variable:
            solver, N = solver_fixed, N_long
        else:
            solver, N = (solver_short, N_short) if mode == "short" else (solver_long, N_long)

        nxu = nx*(N+1) + nu*N
        lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
        ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
        lbx[0] = ubx[0] = float(x)

        if x0 is None or len(x0) != nxu:
            x0 = [float(x)] * nxu

        t0 = time.perf_counter()
        sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N), ubg=[0]*(nx*N), x0=x0)
        T_log.append(time.perf_counter() - t0)

        solx = sol["x"].full().ravel()
        u = ca.DM([solx[nx*(N+1)]])
        x_pred_prev = ca.DM([solx[nx]])
        x0 = np.concatenate([solx[nx:], solx[-nu:]])

        x = rk4(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(s_bar)

    return map(np.array, (X_log, U_log, N_log, S_log, T_log))

# ==============================
# 実行
# ==============================
Xf, Uf, Nf, Sf, Tf = simulate(variable=False)
Xv, Uv, Nv, Sv, Tv = simulate(variable=True)

# ==============================
# ★計算時間の数値評価
# ==============================
print("Fixed mean:", np.mean(Tf))
print("Variable mean:", np.mean(Tv))
print("95% quantile fixed:", np.quantile(Tf, 0.95))
print("95% quantile variable:", np.quantile(Tv, 0.95))

# ==============================
# プロット（a〜e）
# ==============================
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t_grid, Xf, '--', label="Fixed")
axs[0].plot(t_grid, Xv, '-', label="Variable")
axs[0].axhline(h_ref, ls=":")
axs[0].set_title("(a) Water Level")

axs[1].plot(t_grid, Uf, '--')
axs[1].plot(t_grid, Uv, '-')
axs[1].set_title("(b) Control Input")

axs[2].step(t_grid, Nf, '--')
axs[2].step(t_grid, Nv, '-')
axs[2].set_title("(c) Horizon Switching")

axs[3].plot(t_grid, Sf, '--')
axs[3].plot(t_grid, Sv, '-')
axs[3].set_title("(d) Slack Indicator (EMA)")

axs[4].plot(t_grid, Tf, '--')
axs[4].plot(t_grid, Tv, '-')
axs[4].set_title("(e) MPC Computation Time")

for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()
