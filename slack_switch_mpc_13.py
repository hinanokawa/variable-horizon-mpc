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
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])

N_long  = 15
N_short = 5

s_high = 5e-3
s_low  = 1e-4
min_dwell = 5

h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# ==============================
def disturbance(t):
    return -1.0 if 20.0 <= t <= 25.0 else 0.0

# ==============================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")
    h = ca.fmax(x[0], 0.0)
    qi = ca.fmax(u[0] + w, 0.0)
    h_dot = -(S_out/A)*ca.sqrt(2*g*h) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

def rk4(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return ca.fmax(x + dt*(k1+2*k2+2*k3+k4)/6, h_min)

# ==============================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]
    J, G = 0, []

    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])/2
        J += ca.mtimes([U[k].T, R, U[k]])/2
        G.append(X[k+1] - X[k] - dt*f(X[k], U[k], 0))

    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])/2

    return ca.nlpsol(
        "solver", "ipopt",
        {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)},
        {"ipopt.print_level": 0, "print_time": 0}
    )

solver_long  = make_solver(N_long)
solver_short = make_solver(N_short)

# ==============================
def simulate(variable=True):
    x = ca.DM([0.8])
    x_pred_prev = x
    mode = "long"
    dwell = 0

    X_log, U_log, N_log, S_log, T_log = [], [], [], [], []

    for t in t_grid:
        w = disturbance(t)
        s = abs(float(x - x_pred_prev))

        if variable and dwell == 0:
            if mode != "short" and s > s_high:
                mode, dwell = "short", min_dwell
            elif mode != "long" and s < s_low:
                mode, dwell = "long", min_dwell

        dwell = max(dwell-1, 0)
        if not variable:
            mode = "long"

        if mode == "short":
            solver, N = solver_short, N_short
        else:
            solver, N = solver_long, N_long

        nxu = nx*(N+1) + nu*N
        lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
        ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
        lbx[0] = ubx[0] = float(x)

        # ===== 計算時間計測 =====
        t0 = time.perf_counter()
        sol = solver(
            lbx=lbx, ubx=ubx,
            lbg=[0]*(nx*N), ubg=[0]*(nx*N),
            x0=[float(x)]*nxu
        )
        t1 = time.perf_counter()
        T_log.append(t1 - t0)
        # ======================

        solx = sol["x"].full().ravel()
        u = ca.DM([solx[nx*(N+1)]])
        x_pred_prev = ca.DM([solx[nx]])

        x = rk4(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(s)

    return (np.array(X_log), np.array(U_log),
            np.array(N_log), np.array(S_log),
            np.array(T_log))

# ==============================
Xf, Uf, Nf, Sf, Tf = simulate(variable=False)
Xv, Uv, Nv, Sv, Tv = simulate(variable=True)

# ==============================
fig, axs = plt.subplots(5,1, figsize=(10,10), sharex=True)

axs[0].plot(t_grid, Xf, "--", label="Fixed MPC")
axs[0].plot(t_grid, Xv, label="Variable MPC")
axs[0].axhline(h_ref, color="gray", linestyle=":")
axs[0].set_ylabel("h"); axs[0].legend()

axs[1].step(t_grid, Uf, "--", where="post", label="Fixed")
axs[1].step(t_grid, Uv, where="post", label="Variable")
axs[1].set_ylabel("u"); axs[1].legend()

axs[2].step(t_grid, Nv, where="post")
axs[2].set_ylabel("Horizon N")

axs[3].plot(t_grid, Sf)
axs[3].set_ylabel("Slack |x - x̂|")

axs[4].plot(t_grid, Tf*1000, "--", label="Fixed")
axs[4].plot(t_grid, Tv*1000, label="Variable")
axs[4].set_ylabel("Solve time [ms]")
axs[4].set_xlabel("Time [s]")
axs[4].legend()

plt.tight_layout()
plt.show()
