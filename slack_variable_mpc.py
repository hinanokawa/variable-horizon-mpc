import os
import numpy as np
import casadi
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# -----------------------------------------
# パラメータ
# -----------------------------------------
A = 1.0
S_out = 0.1
g = 9.81

nx = 1
nu = 1

Q  = casadi.diag([10.0])
Qf = casadi.diag([200.0])
R  = casadi.diag([0.1])

T = 1.0
K_max = 10   # 最大ホライズン
dt = T / K_max

# スラック関連
lambda_s = 1.0

# 制約
x_lb = [0.01]
x_ub = [1.0]
u_lb = [0.0]
u_ub = [10.0]

h_ref = 0.8
x_ref = casadi.DM([h_ref])
u_ref = casadi.DM([0.0])

# 外乱（可視化のため大きめ）
def disturbance(t):
    if 20 <= t < 25:
        return -1.5
    return 0.0

# 連続モデル（PDF 準拠）
def make_f_cont():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    h = x[0]
    qi = u[0]
    h_pos = casadi.fmax(h, 0)
    hdot = (qi - S_out * casadi.sqrt(2 * g * h_pos)) / A
    return casadi.Function("f_cont", [x, u], [casadi.vertcat(hdot)], ["x", "u"], ["xdot"])

f_cont = make_f_cont()

# RK4 step function creator (returns a one-step discretizer)
def make_F_RK4_step():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    k1 = f_cont(x, u)[0]
    k2 = f_cont(x + dt/2 * k1, u)[0]
    k3 = f_cont(x + dt/2 * k2, u)[0]
    k4 = f_cont(x + dt * k3, u)[0]
    x_next = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return casadi.Function("F_RK4_step", [x, u], [x_next], ["x", "u"], ["x_next"])

F_RK4_step = make_F_RK4_step()

# Create discrete step repeated Kc times in constraints by calling F_RK4_step each step.
def make_nlp_with_slack(Kc):
    X = [casadi.SX.sym(f"x{k}", nx) for k in range(Kc+1)]
    U = [casadi.SX.sym(f"u{k}", nu) for k in range(Kc)]
    s = casadi.SX.sym("s", 1)

    G = []
    J = 0
    for k in range(Kc):
        # stage cost
        J += (casadi.dot(Q @ (X[k]-x_ref), (X[k]-x_ref)) + casadi.dot(R @ (U[k]-u_ref), (U[k]-u_ref))) / 2 * dt
        # dynamics constraint: X[k+1] - F_RK4_step(X[k], U[k]) == 0
        G.append(X[k+1] - F_RK4_step(X[k], U[k])[0])   # <-- 位置引数呼び出し + [0]
    # terminal slack equality
    G.append(X[-1] - (x_ref + s))
    # terminal cost + slack penalty
    J += casadi.dot(Qf @ (X[-1]-x_ref), (X[-1]-x_ref)) / 2 + lambda_s * s**2

    nlp = {"x": casadi.vertcat(*X, *U, s), "f": J, "g": casadi.vertcat(*G)}
    solver = casadi.nlpsol("solver_slack", "ipopt", nlp,
                           {'ipopt': {'max_iter':200, 'print_level':0}, 'print_time': False})
    return solver

def compute_optimal_control_slack(Kc, x_init, x0_guess=None):
    S = make_nlp_with_slack(Kc)
    x0_val = float(x_init.full().ravel()[0])
    # bounds
    lbx = []
    ubx = []
    for _ in range(Kc+1):
        lbx += x_lb; ubx += x_ub
    for _ in range(Kc):
        lbx += u_lb; ubx += u_ub
    lbx.append(-1e3); ubx.append(1e3)  # s bounds
    # fix initial state
    lbx[0] = ubx[0] = float(min(max(x0_val, x_lb[0]), x_ub[0]))
    total_vars = nx*(Kc+1) + nu*Kc + 1
    if x0_guess is None:
        x0 = [0.0]*total_vars
        x0[0] = x0_val
    else:
        x0 = list(x0_guess.full().ravel())
        if len(x0) != total_vars:
            x0 = [0.0]*total_vars
            x0[0] = x0_val
    neq = nx*Kc + nx  # Kc dynamics + terminal slack eq
    lbg = [0.0]*neq; ubg = [0.0]*neq
    res = S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
    z = res["x"].full().ravel()
    offset = nx*(Kc+1)
    u0 = z[offset:offset+nu]
    s_val = z[-1]
    return casadi.DM(u0), float(s_val), casadi.DM(z)

# Simple rule to choose horizon from previous slack value
def choose_horizon_from_slack(prev_s, Kmax=K_max):
    if prev_s is None:
        return Kmax
    if prev_s < 0.2:
        return Kmax
    elif prev_s < 0.8:
        return max(3, Kmax//2)
    else:
        return 3

# Simulation
t_final = 60.0
t_eval = np.arange(0.0, t_final, dt)

x_current = casadi.DM([0.2])
X_log = []
U_log = []
K_log = []
S_guess = None
prev_s = None

for t in t_eval:
    Kc = choose_horizon_from_slack(prev_s)
    u_opt, s_val, sol = compute_optimal_control_slack(Kc, x_current, S_guess)
    # apply to plant
    hdot = f_cont(x_current, u_opt)[0]   # <-- 位置呼び出し + [0]
    w = disturbance(t)
    x_next = x_current + dt * (hdot + w / A)
    X_log.append(float(x_next.full().ravel()[0]))
    U_log.append(float(u_opt.full().ravel()[0]))
    K_log.append(Kc)
    x_current = casadi.DM(x_next)
    S_guess = sol
    prev_s = s_val

# Arrays
X_arr = np.array(X_log)[:len(t_eval)]
U_arr = np.array(U_log)[:len(t_eval)]
K_arr = np.array(K_log)[:len(t_eval)]

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(t_eval, X_arr, label="h"); plt.axhline(h_ref, linestyle="--", color="gray", label="h_ref")
plt.title("Water level (slack / variable horizon)"); plt.ylim(0,1.05); plt.legend(); plt.grid(True)

plt.subplot(1,3,2)
plt.step(t_eval, U_arr, where='post', label="q_i"); plt.title("Control input"); plt.ylim(-0.1,10.1); plt.legend(); plt.grid(True)

plt.subplot(1,3,3)
plt.plot(t_eval, K_arr, marker='o'); plt.title("Horizon K"); plt.ylim(0,K_max+1); plt.grid(True)

plt.tight_layout()
plt.savefig("images/tank_mpc_slack_variable_fixedcalls.png")
plt.show()
print("Saved images/tank_mpc_slack_variable_fixedcalls.png")