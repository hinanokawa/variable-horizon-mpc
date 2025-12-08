# 実行前: pip install casadi pillow
import os
import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

os.makedirs('images', exist_ok=True)

# -----------------------------------------
# タンクパラメータとMPC設定
# -----------------------------------------
A = 1.0        # タンク断面積
C = 0.5        # 流出係数
nx = 1         # 状態次元 h
nu = 1         # 入力次元 u

Q  = casadi.diag([10.0])
Qf = casadi.diag([25.0])
R  = casadi.diag([0.1])

T = 1.0
K = 10
dt = T / K
total = nx * (K + 1) + nu * K

# 水位制約と入力制約
x_lb = [0.01]
x_ub = [1.0]
u_lb = [0.0]
u_ub = [10.0]

h_ref = 0.8
x_ref = casadi.DM([h_ref])
u_ref = casadi.DM([0.0])

# -----------------------------------------
# 外乱 (time-dependent)
# 例：20〜30秒の間だけ放水 (-0.5)
# -----------------------------------------
def add_disturbance(t):
    if 20 <= t < 30:
        return -0.5  # 外乱
    else:
        return 0.0

# -----------------------------------------
# 状態方程式 (PDF 2.4 の非線形モデル)
# -----------------------------------------
S_out = 0.1  # 流出口面積 ← 名前変更

def make_f():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)

    h = x[0]
    qi = u[0]

    g = 9.81
    h_pos = casadi.fmax(h, 0)

    # dh/dt = -(S/A)*sqrt(2*g*h) + qi/A
    h_dot = -(S_out/A) * casadi.sqrt(2*g*h_pos) + qi/A

    return casadi.Function("f", [x, u], [casadi.vertcat(h_dot)], ["x", "u"], ["xdot"])


def make_F_RK4():
    f = make_f()
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)

    k1 = f(x=x, u=u)["xdot"]
    k2 = f(x=x + dt/2*k1, u=u)["xdot"]
    k3 = f(x=x + dt/2*k2, u=u)["xdot"]
    k4 = f(x=x + dt*k3, u=u)["xdot"]

    x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return casadi.Function("F_RK4", [x, u], [x_next], ["x", "u"], ["x_next"])

def compute_stage_cost(x, u):
    return (casadi.dot(Q @ (x-x_ref), (x-x_ref)) + casadi.dot(R @ (u-u_ref), (u-u_ref))) / 2

def compute_terminal_cost(x):
    return casadi.dot(Qf @ (x-x_ref), (x-x_ref)) / 2

# -----------------------------------------
# NLP 生成
# -----------------------------------------
def make_nlp():
    F = make_F_RK4()
    X = [casadi.SX.sym(f"x{k}", nx) for k in range(K+1)]
    U = [casadi.SX.sym(f"u{k}", nu) for k in range(K)]
    G = []
    J = 0

    for k in range(K):
        J += compute_stage_cost(X[k], U[k]) * dt
        G.append(X[k+1] - F(x=X[k], u=U[k])["x_next"])

    J += compute_terminal_cost(X[-1])

    nlp = {"x": casadi.vertcat(*X, *U), "f": J, "g": casadi.vertcat(*G)}
    solver = casadi.nlpsol("solver", "ipopt",
                           nlp, {'ipopt': {'max_iter': 100, 'print_level': 0}})
    return solver

# -----------------------------------------
# MPC 実行
# -----------------------------------------
def compute_optimal_control(S, x_init, x0_guess=None):
    x0_val = float(x_init)
    lbx = []
    ubx = []
    for _ in range(K+1):
        lbx += x_lb
        ubx += x_ub
    for _ in range(K):
        lbx += u_lb
        ubx += u_ub

    lbx[0] = ubx[0] = max(min(x0_val, x_ub[0]), x_lb[0])

    if x0_guess is None:
        x0 = [0.0] * total
        x0[0] = x0_val
    else:
        x0 = list(x0_guess.full().ravel())

    res = S(lbx=lbx, ubx=ubx, lbg=[0]*(K*nx), ubg=[0]*(K*nx), x0=x0)

    sol = res["x"].full().ravel()
    u0 = sol[nx*(K+1):nx*(K+1)+nu]
    return casadi.DM(u0), casadi.DM(sol)

# -----------------------------------------
# シミュレーション
# -----------------------------------------
S = make_nlp()
f = make_f()

t_span = [0, 60]
t_eval = np.arange(t_span[0], t_span[1], dt)

x_current = casadi.DM([0.2])
X_log = [x_current.full().ravel()]
U_log = []
x0_guess = casadi.DM.zeros(total)

for t in t_eval:
    u_opt, sol = compute_optimal_control(S, x_current, x0_guess)
    w = add_disturbance(t)
    xdot = f(x=x_current, u=u_opt)["xdot"]
    x_next = x_current + dt * (xdot + w / A)

    X_log.append(x_next.full().ravel())
    U_log.append(u_opt.full().ravel())
    x_current = x_next
    x0_guess = sol

X_arr = np.array(X_log).reshape(-1, nx)[:len(t_eval)]
U_arr = np.array(U_log).reshape(-1, nu)

# -----------------------------------------
# 結果の可視化（改良版）
# -----------------------------------------
plt.figure(figsize=(12,4))

# ---- 左：水位 h(t) ----
plt.subplot(1,2,1)
plt.plot(t_eval, X_arr[:,0], label="Water level h")
plt.axhline(h_ref, linestyle="--", color="gray", label="Reference h_ref")

# 外乱の時間帯をハイライト（20〜30秒）
plt.axvspan(20, 30, color='red', alpha=0.1, label="Disturbance")

plt.ylim(0,1.1)
plt.xlabel("Time t [s]")
plt.ylabel("Water level h [m]")
plt.title("Tank Water Level (MPC)")
plt.legend()

# ---- 右：入力 u(t) ----
plt.subplot(1,2,2)
plt.step(t_eval, U_arr[:,0], where='post', label="Inflow u", linestyle='--')

plt.axvspan(20, 30, color='red', alpha=0.1, label="Disturbance")

plt.ylim(-0.1, 10.1)
plt.xlabel("Time t [s]")
plt.ylabel("Inflow rate u [m³/s]")
plt.title("Control Input (MPC)")
plt.legend()

plt.tight_layout()
plt.savefig("images/tank_mpc_disturbance_results_labeled.png")
plt.show()
