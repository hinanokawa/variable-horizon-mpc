import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

# =========================================
# 基本設定
# =========================================
A = 1.0          # タンク断面積
S_out = 0.1      # 流出口面積
g = 9.81

nx, nu = 1, 1

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([50.0])
R  = ca.diag([0.1])

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long  = 15
N_mid   = 6
N_short = 3

# スラック判定閾値
skip_th    = 0.02
trigger_th = 0.10

# 入力・状態制約
h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# =========================================
# 外乱（流入量変動として扱う）
# =========================================
def disturbance(t):
    if 20 <= t <= 25:
        return -1.0    # 急激な供給減少
    return 0.0

# =========================================
# タンクモデル（PDF準拠）
# =========================================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")   # 外乱（流入変動）

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
    x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return ca.fmax(x_next, h_min)

# =========================================
# MPC NLP生成
# =========================================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]

    J = 0
    G = []

    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])/2
        J += ca.mtimes([U[k].T, R, U[k]])/2
        G.append(X[k+1] - X[k] - dt*f(X[k], U[k], 0))

    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])/2

    opt_vars = ca.vertcat(*X, *U)
    g = ca.vertcat(*G)

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": opt_vars, "f": J, "g": g},
        {"ipopt.print_level": 0, "print_time": 0}
    )
    return solver

# 事前にソルバ生成
solvers = {
    N_long:  make_solver(N_long),
    N_mid:   make_solver(N_mid),
    N_short: make_solver(N_short)
}

# =========================================
# シミュレーション
# =========================================
def simulate(variable=True):
    x = ca.DM([0.8])
    u_prev = ca.DM([0.0])

    X_log, U_log, N_log, solve_log = [], [], [], []

    for t in t_grid:
        slack = abs(float(x - x_ref))

        # --- 可変ホライズン判定 ---
        if variable:
            if slack < skip_th:
                u = u_prev
                N = None
                solved = 0
            else:
                if slack >= trigger_th:
                    N = N_short
                else:
                    N = N_mid

                solver = solvers[N]
                nxu = nx*(N+1) + nu*N

                lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
                ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
                lbx[0] = ubx[0] = float(x)

                sol = solver(
                    lbx=lbx, ubx=ubx,
                    lbg=[0]*(nx*N), ubg=[0]*(nx*N),
                    x0=[float(x)]*nxu
                )
                sol_x = sol["x"].full().ravel()
                u = ca.DM([sol_x[nx*(N+1)]])
                solved = 1
        else:
            N = N_mid
            solver = solvers[N]
            nxu = nx*(N+1) + nu*N

            lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
            ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
            lbx[0] = ubx[0] = float(x)

            sol = solver(
                lbx=lbx, ubx=ubx,
                lbg=[0]*(nx*N), ubg=[0]*(nx*N),
                x0=[float(x)]*nxu
            )
            sol_x = sol["x"].full().ravel()
            u = ca.DM([sol_x[nx*(N+1)]])
            solved = 1

        w = disturbance(t)
        x = rk4_step(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N if N else 0)
        solve_log.append(solved)

        u_prev = u

    return np.array(X_log), np.array(U_log), np.array(N_log), np.array(solve_log)

# 実行
Xf, Uf, Nf, Sf = simulate(variable=False)
Xv, Uv, Nv, Sv = simulate(variable=True)

# =========================================
# 描画
# =========================================
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_grid, Xf, "--", label="Fixed MPC")
axs[0].plot(t_grid, Xv, label="Variable MPC")
axs[0].axhline(h_ref, color="gray", linestyle=":")
axs[0].set_ylabel("Water level h")
axs[0].legend()

axs[1].step(t_grid, Uf, "--", where="post", label="Fixed MPC")
axs[1].step(t_grid, Uv, where="post", label="Variable MPC")
axs[1].set_ylabel("Input u")
axs[1].legend()

axs[2].step(t_grid, Nv, where="post")
axs[2].set_ylabel("Horizon N")

axs[3].scatter(t_grid, Sf*0, s=10, label="Fixed")
axs[3].scatter(t_grid, Sv*1, s=10, label="Variable")
axs[3].set_yticks([0,1])
axs[3].set_yticklabels(["Fixed","Variable"])
axs[3].set_ylabel("MPC solve")
axs[3].legend()

axs[3].set_xlabel("Time [s]")
plt.tight_layout()
plt.savefig("images/fixed_vs_variable_corrected.png")
plt.show()
