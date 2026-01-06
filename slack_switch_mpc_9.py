import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# =========================================
# 基本設定
# =========================================
A = 1.0; S_out = 0.1; g = 9.81
nx, nu = 1, 1
h_ref = 0.8
x_ref = ca.DM([h_ref])

# 重み（電力系統への応用を意識し、少し強めに設定）
Q = ca.diag([20.0]); Qf = ca.diag([100.0]); R = ca.diag([0.1])

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long = 15; N_mid = 8; N_short = 3

# --- スラック判定閾値 (ここを調整) ---
# 小さな変化でも反応するようにし、精度を確保する
skip_th = 0.001     # これ以下の変化なら計算スキップ（計算スパン切り替え）
trigger_th = 0.01   # これ以上の変化なら N_short（ホライズン切り替え）

h_min, h_max = 0.0, 1.2; u_min, u_max = 0.0, 3.0

def disturbance(t):
    if 20 <= t <= 25: return -1.0
    return 0.0

def make_f():
    x, u, w = ca.SX.sym("x"), ca.SX.sym("u"), ca.SX.sym("w")
    h_dot = -(S_out/A) * ca.sqrt(2*g*ca.fmax(x, 0)) + (ca.fmax(u + w, 0))/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

def rk4_step(x, u, w):
    k1 = f(x, u, w); k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w); k4 = f(x + dt*k3, u, w)
    return ca.fmax(x + dt*(k1 + 2*k2 + 2*k3 + k4)/6, h_min)

def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]
    J = sum(ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)]) + ca.mtimes([U[k].T, R, U[k]]) for k in range(N))/2
    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])/2
    G = [X[k+1] - X[k] - dt*f(X[k], U[k], 0) for k in range(N)]
    return ca.nlpsol("solver", "ipopt", {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)}, 
                     {"ipopt.print_level": 0, "print_time": 0})

solvers = {N_long: make_solver(N_long), N_mid: make_solver(N_mid), N_short: make_solver(N_short)}

def simulate(variable=True):
    x = ca.DM([0.8]); x_prev = x; u = ca.DM([0.28]); u_prev = u
    X_log, U_log, N_log, solve_log = [], [], [], []

    for t in t_grid:
        # スラック変数：前回からの状態変化量
        slack = float(abs(x - x_prev))
        solved = 0
        N_current = N_long # デフォルト

        if variable:
            if slack < skip_th:
                # 【計算スパン切り替え】変化が小さいので前回値を流用
                u = u_prev
                N_plot = 0 # グラフ表示用
            else:
                # 【ホライズン切り替え】変化の大きさに応じて選択
                N_current = N_short if slack >= trigger_th else N_mid
                N_plot = N_current
                solver = solvers[N_current]
                lbx = [h_min]*(nx*(N_current+1)) + [u_min]*(nu*N_current)
                ubx = [h_max]*(nx*(N_current+1)) + [u_max]*(nu*N_current)
                lbx[0] = ubx[0] = float(x)
                sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N_current), ubg=[0]*(nx*N_current), x0=[float(x)]*(nx*(N_current+1)+nu*N_current))
                u = ca.DM([sol["x"].full().ravel()[nx*(N_current+1)]])
                solved = 1
        else:
            N_current = N_long; N_plot = N_current
            solver = solvers[N_current]
            lbx = [h_min]*(nx*(N_current+1)) + [u_min]*(nu*N_current)
            ubx = [h_max]*(nx*(N_current+1)) + [u_max]*(nu*N_current)
            lbx[0] = ubx[0] = float(x)
            sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N_current), ubg=[0]*(nx*N_current), x0=[float(x)]*(nx*(N_current+1)+nu*N_current))
            u = ca.DM([sol["x"].full().ravel()[nx*(N_current+1)]])
            solved = 1

        x_prev = x
        x = rk4_step(x, u, disturbance(t))
        u_prev = u
        X_log.append(float(x)); U_log.append(float(u)); N_log.append(N_plot); solve_log.append(solved)

    return np.array(X_log), np.array(U_log), np.array(N_log), np.array(solve_log)

Xf, Uf, Nf, Sf = simulate(variable=False)
Xv, Uv, Nv, Sv = simulate(variable=True)

# 描画 (元のコードのスタイルを維持)
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t_grid, Xf, "--", label="Fixed MPC"); axs[0].plot(t_grid, Xv, label="Variable MPC")
axs[0].axhline(h_ref, color="gray", linestyle=":"); axs[0].set_ylabel("Level h"); axs[0].legend()
axs[1].step(t_grid, Uf, "--", where="post"); axs[1].step(t_grid, Uv, where="post"); axs[1].set_ylabel("Input u")
axs[2].step(t_grid, Nv, where="post", color="green"); axs[2].set_ylabel("Horizon N")
axs[3].scatter(t_grid, Sv, s=10, color="purple"); axs[3].set_yticks([0,1]); axs[3].set_yticklabels(["Skip","Solve"]); axs[3].set_ylabel("MPC solve")
plt.tight_layout(); plt.show()