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

# 重み行列の調整
Q  = ca.diag([20.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])    # 入力のバタつきを抑えるため少し大きく設定

dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long  = 20  # 安定時
N_short = 5   # 急変時

# --- スラック判定閾値 ---
skip_th    = 0.001  # これより変化が小さければ「計算スキップ」
trigger_th = 0.005  # これより変化が大きければ「短ホライズン(急変対応)」

# 入力・状態制約
h_min, h_max = 0.0, 1.5
u_min, u_max = 0.0, 5.0

# =========================================
# タンクモデル（PDF 式 2.4 準拠）
# =========================================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")
    h = ca.fmax(x[0], 1e-6)
    qi = u[0] + w
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
def simulate(mode="variable"):
    x = ca.DM([0.8]) 
    u_prev = ca.DM([S_out * np.sqrt(2 * g * 0.8)]) # 定常入力
    x_prev = x
    
    logs = {"h": [], "u": [], "N": [], "slack": [], "calc": []}
    calc_count = 0

    for t in t_grid:
        w = -1.5 if 20 <= t <= 25 else 0.0
        
        # スラック変数（状態の変化量）
        slack = float(abs(x - x_prev))
        
        do_calc = True
        N = N_long

        if mode == "variable":
            if slack < skip_th:
                # 【新機能】変化が極小なら前回の入力を使い回し、計算をスキップ
                do_calc = False
                u = u_prev
                N = 0 # グラフ表示用
            elif slack > trigger_th:
                # 急変時は短ホライズン
                N = N_short
            else:
                # それ以外は長ホライズン
                N = N_long
        else:
            # 固定MPCモード
            N = N_long

        if do_calc:
            solver = solvers[N]
            nxu = nx*(N+1) + nu*N
            lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
            ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
            lbx[0] = ubx[0] = float(x)
            
            sol = solver(lbx=lbx, ubx=ubx, lbg=[0]*(nx*N), ubg=[0]*(nx*N), x0=[float(x)]*nxu)
            u = sol["x"].full().ravel()[nx*(N+1)]
            calc_count += 1
        
        logs["h"].append(float(x))
        logs["u"].append(float(u))
        logs["N"].append(N)
        logs["slack"].append(slack)
        logs["calc"].append(1 if do_calc else 0)

        x_prev = x
        u_prev = u
        x = rk4_step(x, u, w)

    return logs, calc_count

# 実行
log_f, count_f = simulate(mode="fixed")
log_v, count_v = simulate(mode="variable")

print(f"Fixed MPC Calculation Count: {count_f}")
print(f"Variable MPC Calculation Count: {count_v}")
print(f"Reduction Rate: {(1 - count_v/count_f)*100:.2f}%")

# =========================================
# 描画
# =========================================
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# 1. 水位
axs[0].plot(t_grid, log_f["h"], "--", label=f"Fixed MPC (N={N_long})")
axs[0].plot(t_grid, log_v["h"], label="Proposed (Variable)")
axs[0].axhline(h_ref, color="red", linestyle=":")
axs[0].set_ylabel("Level [m]")
axs[0].legend()

# 2. 入力
axs[1].step(t_grid, log_f["u"], "--", label="Fixed u")
axs[1].step(t_grid, log_v["u"], label="Proposed u")
axs[1].set_ylabel("Inflow u")
axs[1].legend()

# 3. ホライズン長
axs[2].step(t_grid, log_v["N"], color="green", label="Horizon N")
axs[2].set_ylabel("Horizon N")
axs[2].legend()

# 4. 計算実行フラグ（1なら計算、0ならスキップ）
axs[3].scatter(t_grid, log_v["calc"], s=10, color="purple", label="Calc Executed")
axs[3].set_yticks([0, 1])
axs[3].set_yticklabels(["Skip", "Solve"])
axs[3].set_ylabel("MPC Status")
axs[3].set_xlabel("Time [s]")
axs[3].legend()

plt.tight_layout()
plt.show()