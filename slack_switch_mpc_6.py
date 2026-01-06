import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- 基本設定 ---
A, S_out, g = 1.0, 0.1, 9.81
h_ref = 0.8
dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# 閾値の微調整 (ここが差を出すポイント)
skip_th    = 0.0001 # スキップ判定（もっと厳しくして、必要な時は計算させる）
trigger_th = 0.005  # ホライズン切り替え判定
N_long, N_short = 25, 5

def make_solver(N):
    x, u, w = ca.SX.sym("x"), ca.SX.sym("u"), ca.SX.sym("w")
    h_dot = ((u + w) - S_out * ca.sqrt(2 * g * ca.fmax(x, 1e-6))) / A
    f = ca.Function("f", [x, u, w], [x + dt * h_dot]) # RK4より高速な比較用
    X, U = [ca.SX.sym(f"x{k}") for k in range(N+1)], [ca.SX.sym(f"u{k}") for k in range(N)]
    J = sum(10*(X[k]-h_ref)**2 + 0.5*U[k]**2 for k in range(N)) + 100*(X[-1]-h_ref)**2
    G = [X[k+1] - f(X[k], U[k], 0) for k in range(N)]
    return ca.nlpsol("s", "ipopt", {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)}, {"ipopt.print_level": 0, "print_time": 0})

solvers = {N_long: make_solver(N_long), N_short: make_solver(N_short)}

def run_sim(mode):
    x, x_prev, u = 0.8, 0.8, 0.28
    logs = {"h": [], "u": [], "N": [], "calc": []}
    calc_cnt, curr_N = 0, N_long

    for t in t_grid:
        w = -1.5 if 20 <= t <= 25 else 0.0
        slack = abs(x - x_prev)
        
        do_calc = True
        if mode == "variable":
            if slack < skip_th:
                do_calc = False # 安定時はスキップ
            else:
                curr_N = N_short if slack > trigger_th else N_long # 急変時はN=5、復帰中はN=25
        else:
            curr_N = N_long # 固定MPCは常にN=25で毎ステップ計算

        if do_calc:
            sol = solvers[curr_N](lbx=[x]+[0]*(curr_N*2), ubx=[x]+[1.5]*curr_N+[5.0]*curr_N, lbg=[0]*curr_N, ubg=[0]*curr_N)
            u = float(sol["x"].full().ravel()[curr_N+1])
            calc_cnt += 1

        logs["h"].append(x); logs["u"].append(u); logs["N"].append(curr_N); logs["calc"].append(1 if do_calc else 0)
        x_prev, x = x, x + dt * (((u + w) - S_out * np.sqrt(2 * 9.81 * max(x, 1e-6))) / A)
    return logs, calc_cnt

# 比較実行
res_f, cnt_f = run_sim("fixed")
res_v, cnt_v = run_sim("variable")

print(f"固定MPC計算回数: {cnt_f}回 / 提案手法: {cnt_v}回 (削減率: {100*(1-cnt_v/cnt_f):.1f}%)")

# --- 描画 ---
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t_grid, res_f["h"], "k--", label="Fixed (N=25)")
axs[0].plot(t_grid, res_v["h"], "r-", label="Proposed (Variable)")
axs[0].set_ylabel("Level [m]"); axs[0].legend()

axs[1].step(t_grid, res_f["u"], "k--", alpha=0.5, label="Fixed u")
axs[1].step(t_grid, res_v["u"], "r-", label="Proposed u")
axs[1].set_ylabel("Inflow u"); axs[1].legend()

axs[2].step(t_grid, res_v["N"], "g-", label="Active Horizon N")
axs[2].set_ylabel("N"); axs[2].legend()

axs[3].scatter(t_grid, res_v["calc"], c=res_v["calc"], cmap="coolwarm", s=15, label="Solve/Skip")
axs[3].set_yticks([0, 1]); axs[3].set_yticklabels(["Skip", "Solve"]); axs[3].set_ylabel("Status")
plt.tight_layout(); plt.show()