# 可変ホライズンMPCデモ（Slack を用いて必要ならホライズンを伸ばす）
# 実行前: pip install casadi matplotlib numpy

import os
import numpy as np
import casadi
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# -----------------------------------------
# 共通パラメータ
# -----------------------------------------
A = 1.0        # タンク断面積
nx = 1         # 状態次元 h
nu = 1         # 入力次元 u

# コスト重み
Q  = casadi.diag([10.0])
Qf = casadi.diag([200.0])   # 終端重み（固定・可変とも同じに）
R  = casadi.diag([0.1])

T = 1.0
K_base = 10   # 基本のホライズン（秒ではなく discretization のステップ幅数）
dt = T / K_base

# 出口面積
S_out = 0.1

# 水位制約と入力制約
x_lb = [0.01]
x_ub = [1.0]
u_lb = [0.0]
u_ub = [10.0]

h_ref = 0.8
x_ref = casadi.DM([h_ref])
u_ref = casadi.DM([0.0])

# スラック関連
lambda_s = 1.0    # スラックペナルティ（小さいとスラックが使われやすい）
s_thresh = 1e-3   # スラック閾値：これを超えたらホライズン延長判定
K_max = 30        # ホライズン最大値（安全上の上限）
K_min = K_base    # 基本ホライズン

# 強い外乱（可変ホライズンの効果が見えやすいように強め）
def disturbance(t):
    if 20 <= t < 25:
        return -1.5   # 放水（m^3/s の単位で簡易扱い）
    return 0.0

# -----------------------------------------
# 非線形連続モデル f(x,u) : dh/dt
# dh/dt = -(S_out/A) * sqrt(2*g*h) + qi/A
# -----------------------------------------
g = 9.81

def make_f():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    h = x[0]
    qi = u[0]
    h_pos = casadi.fmax(h, 0)
    h_dot = -(S_out/A) * casadi.sqrt(2*g*h_pos) + qi/A
    return casadi.Function("f", [x, u], [casadi.vertcat(h_dot)], ["x", "u"], ["xdot"])

f = make_f()

def make_F_RK4(dt):
    f_local = make_f()
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    k1 = f_local(x=x, u=u)["xdot"]
    k2 = f_local(x=x + dt/2*k1, u=u)["xdot"]
    k3 = f_local(x=x + dt/2*k2, u=u)["xdot"]
    k4 = f_local(x=x + dt*k3, u=u)["xdot"]
    x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return casadi.Function("F_RK4", [x, u], [x_next], ["x", "u"], ["x_next"])

# -----------------------------------------
# コスト関数
# -----------------------------------------
def compute_stage_cost(x, u):
    return (casadi.dot(Q @ (x-x_ref), (x-x_ref)) + casadi.dot(R @ (u-u_ref), (u-u_ref))) / 2

def compute_terminal_cost(x):
    return casadi.dot(Qf @ (x-x_ref), (x-x_ref)) / 2

# -----------------------------------------
# NLP 作成関数（ホライズン K_eff, soft=True でスラックを含む）
# 戻り値: solver, info_dict (contains sizes)
# -----------------------------------------
def make_nlp(K_eff, dt, soft=False):
    F = make_F_RK4(dt)
    X = [casadi.SX.sym(f"x{k}", nx) for k in range(K_eff+1)]
    U = [casadi.SX.sym(f"u{k}", nu) for k in range(K_eff)]
    G = []
    J = 0

    for k in range(K_eff):
        J += compute_stage_cost(X[k], U[k]) * dt
        # dynamics constraints
        G.append(X[k+1] - F(x=X[k], u=U[k])["x_next"])

    if soft:
        # slack variable (scalar)
        s = casadi.SX.sym("s", 1)
        # terminal constraint relaxed: X[-1] = x_ref + s
        G.append(X[-1] - (x_ref + s))
        # terminal cost + slack penalty
        J += compute_terminal_cost(X[-1]) + lambda_s * s**2
        # decision vars: X, U, s
        vars = casadi.vertcat(*X, *U, s)
        # num constraints = (K_eff)*nx + nx = (K_eff+1)*nx
        g = casadi.vertcat(*G)
    else:
        # hard terminal constraint X[-1] = x_ref
        G.append(X[-1] - x_ref)
        J += compute_terminal_cost(X[-1])
        vars = casadi.vertcat(*X, *U)
        g = casadi.vertcat(*G)

    nlp = {"x": vars, "f": J, "g": g}
    solver = casadi.nlpsol("solver", "ipopt", nlp, {'ipopt': {'max_iter': 200, 'print_level': 0}})
    # provide structural info for bounds and extraction
    num_x = vars.size1()
    num_g = g.size1()
    info = {"num_vars": num_x, "num_cons": num_g, "K_eff": K_eff, "soft": soft}
    return solver, info

# -----------------------------------------
# 最適入力計算（可変ホライズン：soft=True を使う）
# 動作:
#  1) K = K_base で解く
#  2) 解の slack s をチェック（soft==True のとき）
#  3) |s| > s_thresh かつ K < K_max -> K += deltaK -> 再構築 -> 再解（ループ）
#  4) 最終的に u0（現在時刻に適用する1ステップ入力）を返す
# -----------------------------------------
def compute_optimal_control_variable_horizon(x_init, x0_guess=None,
                                             K_start=K_base, dt=dt,
                                             K_max_local=K_max, s_thresh_local=s_thresh):
    # start with K_start
    K_eff = K_start
    # keep last solution x0_guess if provided (useful for warmstart)
    while True:
        solver, info = make_nlp(K_eff, dt, soft=True)
        # build bounds for vars
        num_vars = info["num_vars"]
        # calculate expected var counts:
        # vars = (K_eff+1)*nx + K_eff*nu + (1 if soft else 0)
        # prepare x0
        if x0_guess is None:
            x0 = [0.0] * num_vars
            # place initial state at beginning
            x0[0] = float(x_init)
        else:
            # if previous guess shorter/longer, fallback to zeros
            try:
                x0 = list(x0_guess.full().ravel())
                if len(x0) != num_vars:
                    x0 = [0.0] * num_vars
                    x0[0] = float(x_init)
            except:
                x0 = [0.0] * num_vars
                x0[0] = float(x_init)

        # bounds: lbx, ubx
        lbx = []
        ubx = []
        # states bounds for K_eff+1 states
        for _ in range(K_eff+1):
            lbx += x_lb
            ubx += x_ub
        # inputs bounds for K_eff inputs
        for _ in range(K_eff):
            lbx += u_lb
            ubx += u_ub
        # slack bounds
        lbx.append(-1000.0)
        ubx.append(1000.0)

        # fix current initial state
        lbx[0] = ubx[0] = max(min(float(x_init), x_ub[0]), x_lb[0])

        # constraint bounds: lbg = ubg = 0, number = (K_eff+1)*nx
        lbg = [0.0] * ((K_eff+1)*nx)
        ubg = [0.0] * ((K_eff+1)*nx)

        # solve
        try:
            res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
        except RuntimeError as e:
            # solver failure: if K_eff < K_max then try to increase K, else raise
            if K_eff < K_max_local:
                K_eff = min(K_max_local, K_eff + 2)  # extend by 2 steps and retry
                continue
            else:
                raise

        sol = res["x"].full().ravel()
        # extract slack: slack is last element
        s_val = float(sol[-1])
        # extract first input u0
        # offset to first u: (K_eff+1)*nx
        idx_u0 = (K_eff+1)*nx
        u0 = sol[idx_u0: idx_u0 + nu]

        # check slack threshold
        if abs(s_val) > s_thresh_local and K_eff < K_max_local:
            # increase horizon and re-solve
            K_eff = min(K_max_local, K_eff + 2)
            # make a new guess by padding current sol to new size (simple approach)
            # prepare new x0_guess as zeros (it will be reset in next loop)
            x0_guess = None
            continue
        else:
            # accept solution
            return casadi.DM(u0), casadi.DM(sol), K_eff, s_val

# -----------------------------------------
# 固定ホライズン版（比較用）
# -----------------------------------------
def compute_optimal_control_fixed(x_init, x0_guess=None, K_eff=K_base, dt=dt):
    solver, info = make_nlp(K_eff, dt, soft=False)
    num_vars = info["num_vars"]

    if x0_guess is None:
        x0 = [0.0] * num_vars
        x0[0] = float(x_init)
    else:
        try:
            x0 = list(x0_guess.full().ravel())
            if len(x0) != num_vars:
                x0 = [0.0] * num_vars
                x0[0] = float(x_init)
        except:
            x0 = [0.0] * num_vars
            x0[0] = float(x_init)

    lbx = []
    ubx = []
    for _ in range(K_eff+1):
        lbx += x_lb
        ubx += x_ub
    for _ in range(K_eff):
        lbx += u_lb
        ubx += u_ub

    # fix initial state
    lbx[0] = ubx[0] = max(min(float(x_init), x_ub[0]), x_lb[0])

    lbg = [0.0] * ((K_eff+1)*nx)
    ubg = [0.0] * ((K_eff+1)*nx)

    res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
    sol = res["x"].full().ravel()
    idx_u0 = (K_eff+1)*nx
    u0 = sol[idx_u0: idx_u0 + nu]
    return casadi.DM(u0), casadi.DM(sol)

# -----------------------------------------
# シミュレーション：両方式を同じ初期条件・外乱で比較
# -----------------------------------------
t_span = [0, 60]
t_eval = np.arange(t_span[0], t_span[1], dt)

# 初期値
x0 = casadi.DM([0.2])

# 可変ホライズンシミュレーション
x_current = x0
X_log_var = [x_current.full().ravel()]
U_log_var = []
x0_guess_var = None

# 固定ホライズンシミュレーション
x_current_fix = x0
X_log_fix = [x_current_fix.full().ravel()]
U_log_fix = []
x0_guess_fix = None

K_used_history = []

for t in t_eval:
    # --- 可変ホライズン制御（soft slack を使い必要ならホライズン拡張） ---
    u_opt_var, sol_var, K_used, s_val = compute_optimal_control_variable_horizon(
        x_current, x0_guess_var, K_start=K_min, dt=dt, K_max_local=K_max, s_thresh_local=s_thresh
    )
    # apply disturbance and dynamics (Euler for simplicity using f)
    w = disturbance(t)
    xdot = f(x=x_current, u=u_opt_var)["xdot"]
    # Note: disturbance is in m^3/s; original model divides by A inside f already for input qi/A.
    # We apply x_next = x + dt*(xdot + w)  (consistent with earlier code style)
    x_next = x_current + dt * (xdot + w)

    X_log_var.append(x_next.full().ravel())
    U_log_var.append(u_opt_var.full().ravel())
    x_current = x_next
    # no warmstart supply for simplicity
    x0_guess_var = None

    K_used_history.append(K_used)

    # --- 固定ホライズン制御 ---
    u_opt_fix, sol_fix = compute_optimal_control_fixed(x_current_fix, x0_guess_fix, K_eff=K_base, dt=dt)
    w_fix = disturbance(t)
    xdot_fix = f(x=x_current_fix, u=u_opt_fix)["xdot"]
    x_next_fix = x_current_fix + dt * (xdot_fix + w_fix)
    X_log_fix.append(x_next_fix.full().ravel())
    U_log_fix.append(u_opt_fix.full().ravel())
    x_current_fix = x_next_fix
    x0_guess_fix = None

# convert logs to arrays (trim to t_eval length)
X_arr_var = np.array(X_log_var).reshape(-1, nx)[:len(t_eval)]
U_arr_var = np.array(U_log_var).reshape(-1, nu)

X_arr_fix = np.array(X_log_fix).reshape(-1, nx)[:len(t_eval)]
U_arr_fix = np.array(U_log_fix).reshape(-1, nu)

# -----------------------------------------
# プロット（比較）
# -----------------------------------------
plt.figure(figsize=(12,4))

# 水位比較（左）
plt.subplot(1,2,1)
plt.plot(t_eval, X_arr_fix[:,0], label="Fixed-horizon h")
plt.plot(t_eval, X_arr_var[:,0], label="Variable-horizon h")
plt.axhline(h_ref, linestyle="--", color="gray", label="Reference h_ref")
plt.ylim(0,1.1)
plt.xlabel("Time t [s]")
plt.ylabel("Water level h [m]")
plt.title("Water level: Fixed vs Variable Horizon MPC")
plt.legend()

# 入力比較（右）
plt.subplot(1,2,2)
plt.step(t_eval, U_arr_fix[:,0], where='post', label="Fixed-horizon u", linestyle='--')
plt.step(t_eval, U_arr_var[:,0], where='post', label="Variable-horizon u", linestyle='-')
plt.ylim(-0.1, 10.1)
plt.xlabel("Time t [s]")
plt.ylabel("Inflow rate u [m³/s]")
plt.title("Control input: Fixed vs Variable Horizon MPC")
plt.legend()

plt.tight_layout()
plt.savefig("images/compare_fixed_vs_variable_horizon.png")
plt.show()

# -----------------------------------------
# 補足: 可変版で実際に使われた K の履歴を表示
# -----------------------------------------
print("K_used history (variable-horizon) (per step):")
print(K_used_history)
