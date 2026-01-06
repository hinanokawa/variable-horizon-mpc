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

# 重み行列（電力系統への応用を想定し、追従性を重視）
Q  = ca.diag([20.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long  = 15   # 安定時：じっくり予測
N_mid   = 8    # 過渡期：バランス
N_short = 3    # 急変時：高速応答

# スラック判定閾値（状態の変化量 abs(x - x_prev) に基づく）
# 電力系統の急変対応を想定し、変化に敏感に設定
skip_th    = 0.0005  # これ以下なら計算をスキップ
trigger_th = 0.005   # これ以上なら緊急用(N_short)に切り替え

# 入力・状態制約
h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 5.0

# =========================================
# 外乱（流入量変動）
# =========================================
def disturbance(t):
    if 20 <= t <= 25:
        return -1.5    # 急激な電力需要増や供給減を想定
    return 0.0

# =========================================
# タンクモデル（PDF準拠）
# =========================================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h = ca.fmax(x[0], 0.0)
    qi = ca.fmax(u[0] + w, 0.0)

    # dh/dt = -(S/A)*sqrt(2gh) + qi/A
    h_dot = -(S_out/A) * ca.sqrt(2*g*h) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

# =========================================
# RK4 (プラントシミュレーション用)
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
        G.append(X[k+1] - rk4_step(X[k], U[k], 0)) # 予測時は外乱0と仮定

    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])/2

    opt_vars = ca.vertcat(*X, *U)
    g = ca.vertcat(*G)

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": opt_vars, "f": J, "g": g},
        {"ipopt.print_level": 0, "print_time": 0}
    )
    return solver

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
    x_prev = x
    u = ca.DM([S_out * np.sqrt(2*g*0.8)]) # 初期釣り合い入力
    u_prev = u

    X_log, U_log, N_log, solve_log = [], [], [], []

    for t in t_grid:
        # スラック変数：前回からの状態変化量を検知
        slack = float(abs(x - x_prev))

        if variable:
            if slack < skip_th:
                # 【計算スパンの切り替え】安定時は計算スキップ
                u = u_prev
                N_current = 0
                solved = 0
            else:
                # 【ホライズンの切り替え】変化の大きさに応じて選択
                if slack >= trigger_th:
                    N_current = N_short # 急変時は短いホライズンで即応
                else:
                    N_current = N_long  # 緩やかな変化は長いホライズン
                
                solver = solvers[N_current]
                nxu = nx*(N_current+1) + nu*N_current
                lbx = [h_min]*(nx*(N_current+1)) + [u_min]*(nu*N_current)
                ubx = [h_max]*(nx*(N_current+1)) + [u_max]*(nu*N_current)
                lbx[0] = ubx[0] = float(x)

                sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N_current), ubg=[0]*(nx*N_current), x0=[float(x)]*nxu)
                u = ca.DM([sol["x"].full().ravel()[nx*(N_current+1)]])
                solved = 1
        else:
            # 固定MPC (比較用)
            N_current = N_long
            solver = solvers[N_current]
            nxu = nx*(N_current+1) + nu*N_current
            lbx = [h_min]*(nx*(N_current+1)) + [u_min]*(nu*N_current)
            ubx = [h_max]*(nx*(N_current+1)) + [u_max]*(nu*N_current)
            lbx[0] = ubx[0] = float(x)
            sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N_current), ubg=[0]*(nx*N_current), x0=[float(x)]*nxu)
            u = ca.DM([sol["x"].full().ravel()[nx*(N_current+1)]])
            solved = 1

        w = disturbance(t)
        x_next = rk4_step(x, u, w)
        
        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N_current)
        solve_log.append(solved)

        x_prev = x
        x = x_next
        u_prev = u

    return np.array(X_log), np.array(U_log), np.array(N_log), np.array(solve_log)

# 実行と比較
Xf, Uf, Nf, Sf = simulate(variable=False)
Xv, Uv, Nv, Sv = simulate(variable=True)

# =========================================
# 描画
# =========================================
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs[0].plot(t_grid, Xf, "--", label="Fixed MPC (Benchmark)")
axs[0].plot(t_grid, Xv, label="Variable MPC (Proposed)")
axs[0].axhline(h_ref, color="red", linestyle=":")
axs[0].set_ylabel("Level [m]")
axs[0].legend()

axs[1].step(t_grid, Uf, "--", where="post", label="Fixed MPC")
axs[1].step(t_grid, Uv, where="post", label="Variable MPC")
axs[1].set_ylabel("Input u")
axs[1].legend()

axs[2].step(t_grid, Nv, where="post", color="green", label="Active Horizon")
axs[2].set_ylabel("Horizon N")
axs[2].set_ylim(-1, N_long + 2)

axs[3].scatter(t_grid, Sf*0, s=15, label="Fixed", alpha=0.3)
axs[3].scatter(t_grid, Sv*1, s=15, label="Variable", color="purple")
axs[3].set_yticks([0,1])
axs[3].set_yticklabels(["Skip","Solve"])
axs[3].set_ylabel("Calculation")
axs[3].legend()

plt.tight_layout()
plt.show()