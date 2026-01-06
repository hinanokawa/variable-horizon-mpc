import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- パラメータ設定 (PDF準拠) ---
A, S_out, g = 1.0, 0.1, 9.81
nx, nu = 1, 1
h_ref = 0.8
dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

# 閾値の微調整 (ここが重要)
skip_th    = 0.0001 # スキップ判定（もっと厳しくして計算回数を少し増やす）
trigger_th = 0.005  # ホライズン切り替え判定
N_long, N_short = 25, 5

# --- モデル定義 ---
def make_f():
    x, u, w = ca.SX.sym("x"), ca.SX.sym("u"), ca.SX.sym("w")
    h_dot = ((u + w) - S_out * ca.sqrt(2 * g * ca.fmax(x, 1e-6))) / A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

def rk4_step(x, u, w):
    k1 = f(x, u, w); k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w); k4 = f(x + dt*k3, u, w)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]
    J = sum(10*(X[k]-h_ref)**2 + 0.1*U[k]**2 for k in range(N)) + 50*(X[-1]-h_ref)**2
    G = [X[k+1] - rk4_step(X[k], U[k], 0) for k in range(N)]
    return ca.nlpsol("s", "ipopt", {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)}, {"ipopt.print_level": 0, "print_time": 0})

solvers = {N_long: make_solver(N_long), N_short: make_solver(N_short)}

# --- シミュレーション関数 ---
def run_sim(mode):
    x = ca.DM([0.8]); x_prev = x
    u = ca.DM([S_out * np.sqrt(2 * g * 0.8)]) # 初期入力
    logs = {"h": [], "u": [], "N": [], "calc": []}
    calc_cnt = 0
    current_N = N_long

    for t in t_grid:
        w = -1.5 if 20 <= t <= 25 else 0.0
        slack = float(abs(x - x_prev))
        
        do_calc = True
        if mode == "variable":
            if slack < skip_th:
                do_calc = False # スキップ
            else:
                current_N = N_short if slack > trigger_th else N_long
        else:
            current_N = N_long # 固定モード

        if do_calc:
            sol = solvers[current_N](lbx=[float(x)]+( [0]*(nx*current_N + nu*current_N) ), 
                                     ubx=[float(x)]+( [1.5]*(nx*current_N) + [5.0]*(nu*current_N) ),
                                     lbg=[0]*(nx*current_N), ubg=[0]*(nx*current_N))
            u = sol["x"].full().ravel()[nx*(current_N+1)]
            calc_cnt += 1

        logs["h"].append(float(x))
        logs["u"].append(float(u))
        logs["N"].append(current_N)
        logs["calc"].append(1 if do_calc else 0)
        x_prev = x
        x = rk4_step(x, u, w)
    return logs, calc_cnt

# 比較実行
res_f, cnt_f = run_sim("fixed")
res_v, cnt_v = run_sim("variable")

print(f"Fixed: {cnt_f} times, Variable: {cnt_v} times (Reduction: {100*(1-cnt_v/cnt_f):.1f}%)")

# --- 描画 ---
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t_grid, res_f["h"], "k--", label="Fixed N=25 (Benchmark)")
axs[0].plot(t_grid, res_v["h"], "r-", label="Proposed (Slack-based)")
axs[0].set_ylabel("Level [m]"); axs[0].legend()

axs[1].step(t_grid, res_f["u"], "k--", alpha=0.5)
axs[1].step(t_grid, res_v["u"], "r-", label="Input u")
axs[1].set_ylabel("Inflow u"); axs[1].legend()

axs[2].step(t_grid, res_v["N"], "g-", label="Selected Horizon N")
axs[2].set_ylabel("Horizon N"); axs[2].legend()

axs[3].scatter(t_grid, res_v["calc"], c=res_v["calc"], cmap="coolwarm", s=15)
axs[3].set_yticks([0, 1]); axs[3].set_yticklabels(["Skip", "Solve"]); axs[3].set_ylabel("Calculation")
plt.tight_layout(); plt.show()