import os
import numpy as np
import casadi
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# -----------------------------------------
# タンクパラメータとMPC設定（同じ）
# -----------------------------------------
A = 1.0        
C = 0.5        
nx = 1         
nu = 1         

Q  = casadi.diag([10.0])
Qf = casadi.diag([200.0])     # ★同条件（スラック版と同じ）
R  = casadi.diag([0.1])

T = 1.0
K = 10
dt = T / K

# スラックなし → lambda_s 不要（削除）

x_lb = [0.01]
x_ub = [1.0]
u_lb = [0.0]
u_ub = [10.0]

h_ref = 0.8
x_ref = casadi.DM([h_ref])
u_ref = casadi.DM([0.0])

# -----------------------------------------
# 強い外乱（スラック版と統一）
# -----------------------------------------
def disturbance(t):
    if 20 <= t < 25:
        return -1.5
    return 0.0

# -----------------------------------------
# 状態方程式（同じ）
# -----------------------------------------
S_out = 0.1

def make_f():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    h = x[0]
    qi = u[0]

    g = 9.81
    h_pos = casadi.fmax(h, 0)
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
    return (casadi.dot(Q @ (x-x_ref), (x-x_ref)) +
            casadi.dot(R @ (u-u_ref), (u-u_ref))) / 2

def compute_terminal_cost(x):
    return casadi.dot(Qf @ (x-x_ref), (x-x_ref)) / 2

# -----------------------------------------
# ★ スラックなしの NLP 生成（終端制約 = 等式）
# -----------------------------------------
def make_nlp():
    F = make_F_RK4()
    X = [casadi.SX.sym(f"x{k}", nx) for k in range(K+1)]
    U = [casadi.SX.sym(f"u{k}", nu) for k in range(K)]

    G = []  # 制約
    J = 0  # コスト

    for k in range(K):
        J += compute_stage_cost(X[k], U[k]) * dt
        G.append(X[k+1] - F(x=X[k], u=U[k])["x_next"])

    # ★ スラックなし → 終端は equality constraint X[K] = x_ref
    G.append(X[-1] - x_ref)

    # ★ 終端コストのみ（スラックペナルティなし）
    J += compute_terminal_cost(X[-1])

    # ★ s を含まない
    nlp = {
        "x": casadi.vertcat(*X, *U),
        "f": J,
        "g": casadi.vertcat(*G)
    }

    solver = casadi.nlpsol(
        "solver", "ipopt", nlp,
        {'ipopt': {'max_iter': 200, 'print_level': 0}}
    )
    return solver

# -----------------------------------------
# MPC 実行関数（スラック関連削除）
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

    # ★ s が無いので bounds も短い

    # 初期状態固定
    lbx[0] = ubx[0] = max(min(x0_val, x_ub[0]), x_lb[0])

    # 初期推定
    if x0_guess is None:
        total = nx*(K+1) + nu*K
        x0 = [0.0] * total
        x0[0] = x0_val
    else:
        x0 = list(x0_guess.full().ravel())

    # ★ 制約数は K + 1
    lbg = [0]*(K*nx + 1)
    ubg = [0]*(K*nx + 1)

    res = S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
    sol = res["x"].full().ravel()

    u0 = sol[nx*(K+1):nx*(K+1)+nu]
    return casadi.DM(u0), casadi.DM(sol)

# -----------------------------------------
# シミュレーション（同条件）
# -----------------------------------------
S = make_nlp()
f = make_f()

t_span = [0, 60]
t_eval = np.arange(t_span[0], t_span[1], dt)

x_current = casadi.DM([0.2])
X_log = [x_current.full().ravel()]
U_log = []
total_no_slack = nx*(K+1) + nu*K
x0_guess = casadi.DM.zeros(total_no_slack)

for t in t_eval:
    u_opt, sol = compute_optimal_control(S, x_current, x0_guess)

    w = disturbance(t)

    xdot = f(x=x_current, u=u_opt)["xdot"]
    x_next = x_current + dt * (xdot + w)

    X_log.append(x_next.full().ravel())
    U_log.append(u_opt.full().ravel())

    x_current = x_next
    x0_guess = sol

X_arr = np.array(X_log).reshape(-1, nx)[:len(t_eval)]
U_arr = np.array(U_log).reshape(-1, nu)

# -----------------------------------------
# 結果の可視化（スラック版と同じ）
# -----------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(t_eval, X_arr[:,0], label="Water level h")
plt.axhline(h_ref, linestyle="--", color="gray", label="Reference h_ref")
plt.ylim(0,1.1)
plt.xlabel("Time t [s]")
plt.ylabel("Water level h [m]")
plt.title("Tank Water Level (Fixed-Horizon MPC)")
plt.legend()

plt.subplot(1,2,2)
plt.step(t_eval, U_arr[:,0], where='post', label="Inflow u", linestyle='--')
plt.ylim(-0.1, 10.1)
plt.xlabel("Time t [s]")
plt.ylabel("Inflow rate u [m³/s]")
plt.title("Control Input (Fixed-Horizon MPC)")
plt.legend()

plt.tight_layout()
plt.savefig("images/tank_mpc_fixed_horizon.png")
plt.show()
