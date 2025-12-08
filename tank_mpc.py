# 実行前: pip install casadi pillow
import os
import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

os.makedirs('images', exist_ok=True)

# -------------------------
# 5.6.1 定数・制約条件（倒立振子->タンクに差し替え）
# -------------------------
# タンクパラメータ
A = 1.0        # タンク断面積
C = 0.5        # 流出係数（流量 = C * sqrt(h) のような形）
# MPC の次元
nu = 1         # 制御入力次元（注入流量）
nx = 1         # 状態次元（h）

# コスト関数の重み（スカラーを1x1行列で表す）
Q = casadi.diag([10.0])     # 水位誤差重み
Q_f = casadi.diag([20.0])   # 終端重み（少し大きめ）
R = casadi.diag([0.1])      # 入力コスト

# 予測ホライズン等
T = 1.0
K = 10
dt = T / K

# -------------------------
# 制約条件（ここを調整）
# -------------------------
# 下限を 0.01 にして「絶対に 0 にならない」ようにする（枯渇防止）
# 上限を 1.0 にして「溢れない」ようにする
x_lb = [0.01]     # 水位下限（負にならないように0ではなく小さい正数）
x_ub = [1.0]      # 水位上限（タンク容量）
u_lb = [0.0]      # 注入流量下限（負の注入なし）
u_ub = [10.0]     # 注入流量上限（ポンプ能力など）

# 目標値（目的の水位）
h_ref_value = 0.8   # 例: 目標水位 0.8 m（必要なら変更）
x_ref = casadi.DM([h_ref_value])
u_ref = casadi.DM([0.0])

total = nx * (K + 1) + nu * K  # 最適化変数全体の長さ

# -------------------------
# 5.6.2 状態方程式（タンクモデル）
# -------------------------
def make_f():
    states = casadi.SX.sym("states", nx)
    ctrls = casadi.SX.sym("ctrls", nu)

    h = states[0]
    u = ctrls[0]

    # 流出 = C * sqrt(max(h, 0))  (h<0 の sqrt 回避のため fmax を使う)
    h_nonneg = casadi.fmax(h, 0)
    outflow = C * casadi.sqrt(h_nonneg)

    h_dot = (u - outflow) / A

    states_dot = casadi.vertcat(h_dot)

    f = casadi.Function("f", [states, ctrls], [states_dot], ['x', 'u'], ['x_dot'])
    return f

# -------------------------
# 5.6.3 RK4 による時間離散化（そのまま流用）
# -------------------------
def make_F_RK4():
    states = casadi.SX.sym("states", nx)
    ctrls = casadi.SX.sym("ctrls", nu)

    f = make_f()

    r1 = f(x=states, u=ctrls)["x_dot"]
    r2 = f(x=states + dt * r1 / 2, u=ctrls)["x_dot"]
    r3 = f(x=states + dt * r2 / 2, u=ctrls)["x_dot"]
    r4 = f(x=states + dt * r3, u=ctrls)["x_dot"]

    states_next = states + dt * (r1 + 2 * r2 + 2 * r3 + r4) / 6

    F_RK4 = casadi.Function("F_RK4", [states, ctrls], [states_next], ["x", "u"], ["x_next"])
    return F_RK4

# -------------------------
# 5.6.4 積分器の作成（cvodes を使う）
# -------------------------
def make_integrator():
    states = casadi.SX.sym("states", nx)
    ctrls = casadi.SX.sym("ctrls", nu)

    f = make_f()
    ode = f(x=states, u=ctrls)["x_dot"]

    # dae の定義: x'=ode(x,u), p=ctrls をパラメータとして与える
    dae = {"x": states, "p": ctrls, "ode": ode}

    # integrator: ステップ幅 dt
    I = casadi.integrator("I", "cvodes", dae, {"tf": dt})
    return I

# -------------------------
# 5.6.5 評価関数（stage / terminal）
# -------------------------
def compute_stage_cost(x, u):
    x_diff = x - x_ref
    u_diff = u - u_ref
    # (x'Q x + u'R u)/2 の形
    cost = (casadi.dot(Q @ x_diff, x_diff) + casadi.dot(R @ u_diff, u_diff)) / 2
    return cost

def compute_terminal_cost(x):
    x_diff = x - x_ref
    cost = casadi.dot(Q_f @ x_diff, x_diff) / 2
    return cost

# -------------------------
# 5.6.6 最適化問題の定式化（最小限の変更）
# ※ 状態・入力の等式制約（動力学）は G に入れている。
#    初期条件は compute_optimal_control 側で lbx/ubx により固定します。
# -------------------------
def make_nlp():
    F_RK4 = make_F_RK4()

    U = [casadi.SX.sym(f"u_{k}", nu) for k in range(K)]
    X = [casadi.SX.sym(f"x_{k}", nx) for k in range(K + 1)]
    G = []

    J = 0

    for k in range(K):
        J += compute_stage_cost(X[k], U[k]) * dt
        eq = X[k + 1] - F_RK4(x=X[k], u=U[k])["x_next"]
        G.append(eq)
    J += compute_terminal_cost(X[-1])

    # NLP を構築（変数は [X_0,...,X_K, U_0,...,U_{K-1}]）
    nlp = {"x": casadi.vertcat(*X, *U), "f": J, "g": casadi.vertcat(*G)}
    option = {'print_time': False, 'ipopt': {'max_iter': 100, 'print_level': 0}}
    S = casadi.nlpsol("S", "ipopt", nlp, option)
    return S

# -------------------------
# 5.6.7 最適入力を計算する関数
#    - 引数 x_init: 現在の状態（casadi.DM）
#    - x0_guess: 初期推定（初回は zeros でよい）
# 重要: 初期状態 X_0 を x_init に固定するため、
#        lbx/ubx の最初の nx 要素を x_init にする処理を追加。
# -------------------------
def compute_optimal_control(S, x_init, x0_guess=None):
    # x_init: casadi.DM -> python list
    x_init_list = list(x_init.full().ravel())

    # 変数順は [X_0,...,X_K, U_0,...,U_{K-1}]
    # lbx, ubx を作る
    lbx = []
    ubx = []
    # X bounds: for K+1 states
    for i in range(K + 1):
        lbx += x_lb
        ubx += x_ub
    # U bounds: for K controls
    for i in range(K):
        lbx += u_lb
        ubx += u_ub

    # 初期状態 X_0 を観測値で固定（下限=上限=x_init）
    # 先にクリップして制約に収める（万が一観測が制約外なら安全に処理）
    x0_clip = []
    for val, lb, ub in zip(x_init_list, x_lb, x_ub):
        x0_clip.append(float(min(max(val, lb), ub)))
    # 上書き
    for i in range(nx):
        lbx[i] = x0_clip[i]
        ubx[i] = x0_clip[i]

    # 初期解の推定 x0（最適化変数の初期値）
    if x0_guess is None:
        x0 = np.zeros(total).tolist()
        # 初期状態 x0 の最初の nx entries は現在の観測値にする（良い初期化）
        for i in range(nx):
            x0[i] = float(x0_clip[i])
    else:
        # x0_guess が渡されれば使う（casadi.MX/DM など）
        x0 = list(x0_guess.full().ravel())

    # 等式制約の RHS (すべて 0)
    lbg = [0.0] * (nx * K)
    ubg = [0.0] * (nx * K)

    # IPOpt を呼ぶ (lbx/ubx を渡して状態・入力の box 制約を適用)
    res = S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)

    sol = res["x"].full().ravel()
    offset = nx * (K + 1)
    # 最初の制御入力だけ取り出す
    u0 = sol[offset:offset + nu]
    # 全最適解が必要な場合は sol[offset:] を参照する
    return casadi.DM(u0), casadi.DM(sol)

# -------------------------
# 5.6.8 MPC の実行（シミュレーション）
#    - t_span, x_init に注目（x_init は下限/上限に収める）
# -------------------------
# シミュレーション時間（必要に応じて変更可能）
t_span = [0.0, 60.0]
t_eval = np.arange(t_span[0], t_span[1], dt)

# 初期値（例: 水位 0.2m）。下限より低ければ自動クリップされます。
x_init = casadi.DM([0.2])
# 初期の最適化変数推定
x0_guess = casadi.DM.zeros(total)

S = make_nlp()
I = make_integrator()

X_sim = [x_init.full().ravel().copy()]
U_sim = []

x_current = x_init
x0_for_guess = x0_guess

for step, t in enumerate(t_eval):
    # 現在の観測 x_current を元に最適化 → 制御適用 → 状態更新
    u_opt, sol = compute_optimal_control(S, x_current, x0_for_guess)
    # integrator に p=ctrls として渡す
    x_next = I(x0=x_current, p=u_opt)["xf"]
    # ログ
    X_sim.append(x_next.full().ravel().copy())
    U_sim.append(u_opt.full().ravel().copy())
    # 次ステップへ
    x_current = x_next
    # 初期推定を今回の解で更新して次に使う（ウォームスタートの簡易）
    x0_for_guess = sol

# 整形（t_eval と同じ長さに揃える）
X_arr = np.array(X_sim).reshape(-1, nx)[:len(t_eval)]
U_arr = np.array(U_sim).reshape(-1, nu)

# -------------------------
# 結果の可視化
# -------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_eval, X_arr[:, 0], label="h (water level)")
plt.axhline(h_ref_value, color='gray', linestyle='--', label='h_ref')
plt.ylim(-0.05, max(1.05, float(x_ub[0]) + 0.05))
plt.xlabel("Time [s]")
plt.ylabel("Water level h")
plt.legend()

plt.subplot(1, 2, 2)
plt.step(t_eval, U_arr[:, 0], where='post', linestyle='--', label="u (inflow)")
plt.ylim(min(0.0, float(u_lb[0]) - 0.1), float(u_ub[0]) + 0.1)
plt.xlabel("Time [s]")
plt.ylabel("Control u")
plt.legend()

plt.tight_layout()
plt.savefig("images/tank_mpc_results.png")
plt.show()

# -------------------------
# 5.6.9 アニメーション（簡易：タンクを長方形で表示し、水位を矩形で塗る）
# -------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
frames = np.arange(0, len(t_eval))
fps = int(1 / dt) if dt > 0 else 10

tank_width = 1.0
tank_height_px = 4.0  # 描画上の高さスケール

def update_figure(i):
    ax.cla()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, tank_height_px)
    ax.set_aspect("equal")
    ax.set_title(f"t={t_eval[i]:0.2f} s")

    # 描画スケール：実際の水位 h をプロット上の高さに変換
    h = float(X_arr[i, 0])
    # スケール係数（見やすさのため）
    scale = 2.0
    h_plot = min(h * scale, tank_height_px)

    # タンク外枠
    rect = patches.Rectangle((-tank_width/2, 0), tank_width, tank_height_px, fill=False, linewidth=2)
    ax.add_patch(rect)

    # 水面（塗りつぶし）
    water = patches.Rectangle((-tank_width/2, 0), tank_width, h_plot, facecolor='blue', alpha=0.6)
    ax.add_patch(water)

    ax.text(-0.9, tank_height_px - 0.3, f"h = {h:.3f} [unit]", fontsize=10)
    # 現在の注入流量を表示
    u = float(U_arr[i, 0])
    ax.text(-0.9, tank_height_px - 0.6, f"u = {u:.3f}", fontsize=10)

ani = FuncAnimation(fig, update_figure, frames=frames, interval=1000*dt)
# GIF 保存（任意、時間がかかることがあります）
ani.save("images/tank_mpc_animation.gif", writer="pillow", fps=fps)
print("Saved images/tank_mpc_results.png and images/tank_mpc_animation.gif")
