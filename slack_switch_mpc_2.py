import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# =========================================
# 基本設定 (PDFのパラメータに準拠)
# =========================================
A = 1.0          # タンク断面積 [m^2]
S_out = 0.1      # 流出口面積 [m^2]
g = 9.81         # 重力加速度 [m/s^2]

nx, nu = 1, 1
h_ref = 0.8
x_ref = ca.DM([h_ref])

# 重み行列の調整（応答を見やすくするため）
Q  = ca.diag([20.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.01])

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long  = 30  # 安定時：じっくり先を読む
N_short = 5   # 急変時：手短に計算して即応する

# スラック判定閾値（状態の変化量 |h_k - h_{k-1}|）
trigger_th = 0.005 # これを超えたら「急変」とみなす

# 入力・状態制約
h_min, h_max = 0.0, 1.5
u_min, u_max = 0.0, 5.0

# =========================================
# タンクモデル（PDF 式 2.4 準拠）
# =========================================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")  # 外乱（流入変動）

    h = ca.fmax(x[0], 1e-6) # 平方根内の負値を防ぐ
    qi = u[0] + w

    # PDFの式: dh/dt = (qi - S*sqrt(2gh)) / A
    h_dot = (qi - S_out * ca.sqrt(2 * g * h)) / A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

def rk4_step(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# =========================================
# MPC NLP生成
# =========================================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]
    J = 0
    G = []
    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])
        J += ca.mtimes([U[k].T, R, U[k]])
        # 予測モデルには外乱w=0を想定
        G.append(X[k+1] - rk4_step(X[k], U[k], 0))
    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])
    
    opt_vars = ca.vertcat(*X, *U)
    g = ca.vertcat(*G)
    return ca.nlpsol("solver", "ipopt", {"x": opt_vars, "f": J, "g": g}, 
                     {"ipopt.print_level": 0, "print_time": 0})

solvers = {N_long: make_solver(N_long), N_short: make_solver(N_short)}

# =========================================
# シミュレーション実行
# =========================================
def simulate(variable=True):
    x = ca.DM([0.8]) # 初期値（目標値で安定している状態）
    u_prev = ca.DM([S_out * np.sqrt(2 * g * 0.8)]) # 定常入力
    x_prev = x

    logs = {"h": [], "u": [], "N": [], "slack": []}

    for i, t in enumerate(t_grid):
        # 外乱の設定
        w = -1.5 if 20 <= t <= 25 else 0.0
        
        # スラック変数の計算（前時刻からの水位の変化量）
        slack = float(abs(x - x_prev))
        
        # ホライズン決定ロジック
        if variable:
            N = N_short if slack > trigger_th else N_long
        else:
            N = N_long # 比較用の固定MPC

        # 最適化実行
        solver = solvers[N]
        nxu = nx*(N+1) + nu*N
        lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
        ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
        lbx[0] = ubx[0] = float(x)
        
        sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N), ubg=[0]*(nx*N), x0=[float(x)]*nxu)
        u = sol["x"].full().ravel()[nx*(N+1)] # 最初の入力を採用
        
        # ログ記録
        logs["h"].append(float(x))
        logs["u"].append(float(u))
        logs["N"].append(N)
        logs["slack"].append(slack)

        # 状態更新
        x_prev = x
        x = rk4_step(x, u, w)

    return logs

# 実行
log_f = simulate(variable=False)
log_v = simulate(variable=True)

# =========================================
# 結果の可視化
# =========================================
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_grid, log_f["h"], "--", label="Fixed Horizon (N=30)")
axs[0].plot(t_grid, log_v["h"], label="Variable Horizon (Proposed)")
axs[0].axhline(h_ref, color="red", linestyle=":", label="Target")
axs[0].set_ylabel("Water level [m]")
axs[0].legend()
axs[0].set_title("Tank Level Control with Variable Horizon MPC")

axs[1].step(t_grid, log_v["u"], label="Input u (Variable)", color="orange")
axs[1].set_ylabel("Inflow u [m^3/s]")
axs[1].legend()

axs[2].step(t_grid, log_v["N"], label="Horizon N", color="green")
axs[2].set_ylabel("Horizon Length")
axs[2].set_xlabel("Time [s]")
axs[2].legend()

plt.tight_layout()
plt.show()