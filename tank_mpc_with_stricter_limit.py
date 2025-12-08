# 実行前: pip install casadi pillow
import os
import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

os.makedirs('images', exist_ok=True)

# -------------------------
# タンク制御の定数・制約条件
# -------------------------
A = 1.0
C = 0.5
nu = 1
nx = 1

Q = casadi.diag([10.0])
Q_f = casadi.diag([20.0])
R = casadi.diag([0.1])

T = 1.0
K = 10
dt = T / K

# ---------- ここを変更 ----------
x_lb = [0.01]   # 水位下限（枯渇防止）
x_ub = [1.0]    # 水位上限（溢れ防止）
u_lb = [0.0]    # 注入下限
u_ub = [1.0]    # ← **厳しくした入力上限（ポンプの最大流量）**
# --------------------------------

h_ref_value = 0.8
x_ref = casadi.DM([h_ref_value])
u_ref = casadi.DM([0.0])

total = nx * (K + 1) + nu * K

# -------------------------
# タンクモデル
# -------------------------
def make_f():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)

    h = x[0]
    inflow = u[0]

    outflow = C * casadi.sqrt(casadi.fmax(h, 0))

    h_dot = (inflow - outflow) / A
    return casadi.Function("f", [x, u], [casadi.vertcat(h_dot)], ["x", "u"], ["x_dot"])

# -------------------------
# RK4
# -------------------------
def make_F_RK4():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)

    f = make_f()

    k1 = f(x, u)
    k2 = f(x + dt * k1 / 2, u)
    k3 = f(x + dt * k2 / 2, u)
    k4 = f(x + dt * k3, u)

    x_next = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return casadi.Function("F_RK4", [x, u], [x_next], ["x", "u"], ["x_next"])

# -------------------------
# Integrator (cvodes)
# -------------------------
def make_integrator():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)

    f = make_f()
    ode = f(x, u)

    dae = {"x": x, "p": u, "ode": ode}
    return casadi.integrator("I", "cvodes", dae, {"tf": dt})

# -------------------------
# コスト関数
# -------------------------
def compute_stage_cost(x, u):
    return (casadi.dot(Q @ (x - x_ref), (x - x_ref)) + casadi.dot(R @ (u - u_ref), (u - u_ref))) / 2

def compute_terminal_cost(x):
    return casadi.dot(Q_f @ (x - x_ref), (x - x_ref)) / 2

# -------------------------
# NLP
# -------------------------
def make_nlp():
    F = make_F_RK4()

    U = [casadi.SX.sym(f"u_{k}", nu) for k in range(K)]
    X = [casadi.SX.sym(f"x_{k}", nx) for k in range(K+1)]
    G = []
    J = 0

    for k in range(K):
        J += compute_stage_cost(X[k], U[k]) * dt
        G.append(X[k+1] - F(x=X[k], u=U[k])["x_next"])

    J += compute_terminal_cost(X[-1])

    nlp = {"x": casadi.vertcat(*X, *U), "f": J, "g": casadi.vertcat(*G)}
    S = casadi.nlpsol("S", "ipopt", nlp)
    return S

# -------------------------
# 最適入力を計算
# -------------------------
def compute_optimal_control(S, x_init, x0_guess=None):

    x_init_list = list(x_init.full().ravel())

    lbx = []
    ubx = []
    for _ in range(K + 1):
        lbx += x_lb
        ubx += x_ub
    for _ in range(K):
        lbx += u_lb
        ubx += u_ub

    # 初期状態 X_0 を観測値で固定（下限=上限=x_init）
    for i in range(nx):
        lbx[i] = ubx[i] = float(min(max(x_init_list[i], x_lb[i]), x_ub[i]))

    if x0_guess is None:
        x0 = np.zeros(total).tolist()
        x0[0] = float(x_init_list[0])
    else:
        x0 = list(x0_guess.full().ravel())

    res = S(lbx=lbx, ubx=ubx, lbg=[0]*(nx*K), ubg=[0]*(nx*K), x0=x0)

    sol = res["x"].full().ravel()
    offset = nx * (K + 1)
    u0 = sol[offset:offset + nu]
    return casadi.DM(u0), casadi.DM(sol)

# -------------------------
# MPC 実行
# -------------------------
t_span = [0.0, 60.0]
t_eval = np.arange(t_span[0], t_span[1], dt)

# ---------- ここを変更 ----------
x_init = casadi.DM([0.1])   # 初期をさらに低くして差をわかりやすくする
# --------------------------------

x_guess = casadi.DM.zeros(total)

S = make_nlp()
I = make_integrator()

X_sim = [x_init.full().ravel().copy()]
U_sim = []

x_current = x_init
x0_for_guess = x_guess

for t in t_eval:
    u_opt, sol = compute_optimal_control(S, x_current, x0_for_guess)
    x_next = I(x0=x_current, p=u_opt)["xf"]
    X_sim.append(x_next.full().ravel())
    U_sim.append(u_opt.full().ravel())
    x_current = x_next
    x0_for_guess = sol

X_arr = np.array(X_sim).reshape(-1, nx)[:len(t_eval)]
U_arr = np.array(U_sim).reshape(-1, nu)

# -------------------------
# 結果の可視化（見やすい元の形式へ統一）
# -------------------------
plt.figure(figsize=(12,4))

# 水位
plt.subplot(1, 2, 1)
plt.plot(t_eval, X_arr[:,0], label="h (water level)")
plt.axhline(h_ref_value, linestyle="--", label="h_ref")
plt.xlabel("Time [s]")
plt.ylabel("Water Level h [m]")
plt.legend(loc="best")
plt.grid(True)

# 入力
plt.subplot(1, 2, 2)
plt.step(t_eval, U_arr[:,0], where="post", linestyle="--", label="u (inflow)")
plt.xlabel("Time [s]")
plt.ylabel("Control u")
plt.legend(loc="best")
plt.grid(True)

plt.tight_layout()
plt.savefig("images/tank_mpc_with_stricter_limit.png")
plt.show()
