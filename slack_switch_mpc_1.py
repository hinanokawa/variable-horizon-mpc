import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

# =========================================
# パラメータ設定
# =========================================
A = 1.0
S_out = 0.1
g = 9.81

nx, nu = 1, 1

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])

rho_s = 50.0      # ★スラック重み（重要）

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long  = 15
N_mid   = 6
N_short = 3

# スラック判定閾値
s_skip    = 0.01
s_trigger = 0.08

# 制約
h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# =========================================
# 外乱
# =========================================
def disturbance(t):
    if 20 <= t <= 25:
        return -1.0
    return 0.0

# =========================================
# タンクモデル
# =========================================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h = ca.fmax(x[0], 0.0)
    qi = ca.fmax(u[0] + w, 0.0)

    h_dot = -(S_out/A) * ca.sqrt(2*g*h) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

# =========================================
# RK4
# =========================================
def rk4_step(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return ca.fmax(x + dt*(k1 + 2*k2 + 2*k3 + k4)/6, h_min)

# =========================================
# 終端スラック付き MPC ソルバ生成
# =========================================
def make_solver_with_slack(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]
    s = ca.SX.sym("s", 1)  # ★終端スラック

    J = 0
    G = []

    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)]) / 2
        J += ca.mtimes([U[k].T, R, U[k]]) / 2
        G.append(X[k+1] - X[k] - dt*f(X[k], U[k], 0))

    # ★終端スラック制約
    G.append(X[-1] - (x_ref + s))

    # ★終端コスト + スラックペナルティ
    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)]) / 2
    J += rho_s * s**2

    opt_vars = ca.vertcat(*X, *U, s)
    g = ca.vertcat(*G)

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": opt_vars, "f": J, "g": g},
        {"ipopt.print_level": 0, "print_time": 0}
    )
    return solver

# 事前生成
solvers = {
    N_long:  make_solver_with_slack(N_long),
    N_mid:   make_solver_with_slack(N_mid),
    N_short: make_solver_with_slack(N_short)
}

# =========================================
# シミュレーション
# =========================================
def simulate(variable=True):
    x = ca.DM([0.8])
    u_prev = ca.DM([0.0])

    X_log, U_log, N_log, S_log, solve_log = [], [], [], [], []

    for t in t_grid:
        if variable:
            s_est = abs(float(x - x_ref))

            if s_est < s_skip:
                u = u_prev
                N = 0
                s_val = 0.0
                solved = 0
            else:
                N = N_short if s_est > s_trigger else N_mid
                solver = solvers[N]

                nv = nx*(N+1) + nu*N + 1
                lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N) + [-1.0]
                ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N) + [ 1.0]
                lbx[0] = ubx[0] = float(x)

                sol = solver(
                    lbx=lbx, ubx=ubx,
                    lbg=[0]*(nx*N + 1),
                    ubg=[0]*(nx*N + 1),
                    x0=[float(x)]*nv
                )

                sol_x = sol["x"].full().ravel()
                u = ca.DM([sol_x[nx*(N+1)]])
                s_val = sol_x[-1]
                solved = 1
        else:
            N = N_mid
            solver = solvers[N]

            nv = nx*(N+1) + nu*N + 1
            lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N) + [-1.0]
            ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N) + [ 1.0]
            lbx[0] = ubx[0] = float(x)

            sol = solver(
                lbx=lbx, ubx=ubx,
                lbg=[0]*(nx*N + 1),
                ubg=[0]*(nx*N + 1),
                x0=[float(x)]*nv
            )

            sol_x = sol["x"].full().ravel()
            u = ca.DM([sol_x[nx*(N+1)]])
            s_val = sol_x[-1]
            solved = 1

        w = disturbance(t)
        x = rk4_step(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(abs(s_val))
        solve_log.append(solved)

        u_prev = u

    return map(np.array, (X_log, U_log, N_log, S_log, solve_log))

# 実行
Xf, Uf, Nf, Sf, _ = simulate(variable=False)
Xv, Uv, Nv, Sv, _ = simulate(variable=True)

# =========================================
# 描画
# =========================================
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

axs[0].plot(t_grid, Xf, "--", label="Fixed MPC")
axs[0].plot(t_grid, Xv, label="Variable MPC")
axs[0].axhline(h_ref, linestyle=":", color="gray")
axs[0].set_ylabel("Water level h")
axs[0].legend()

axs[1].step(t_grid, Uf, "--", where="post", label="Fixed MPC")
axs[1].step(t_grid, Uv, where="post", label="Variable MPC")
axs[1].set_ylabel("Input u")
axs[1].legend()

axs[2].step(t_grid, Nv, where="post")
axs[2].set_ylabel("Horizon N")

axs[3].plot(t_grid, Sv)
axs[3].set_ylabel("Terminal slack |s|")
axs[3].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("images/slack_based_variable_mpc.png")
plt.show()
