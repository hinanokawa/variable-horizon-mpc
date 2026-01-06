import os
import numpy as np
import casadi
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# -----------------------------------------
# パラメータ（同じに合わせる）
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
K = 10
dt = T / K

# 制約
x_lb = [0.01]
x_ub = [1.0]
u_lb = [0.0]
u_ub = [10.0]

h_ref = 0.8
x_ref = casadi.DM([h_ref])
u_ref = casadi.DM([0.0])

# disturbance same as slack version
def disturbance(t):
    if 20 <= t < 25:
        return -1.5
    return 0.0

# continuous model
def make_f_cont():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    h = x[0]; qi = u[0]
    h_pos = casadi.fmax(h, 0)
    hdot = (qi - S_out * casadi.sqrt(2 * g * h_pos)) / A
    return casadi.Function("f_cont", [x, u], [casadi.vertcat(hdot)], ["x", "u"], ["xdot"])

f_cont = make_f_cont()

# RK4 one-step function (use positional calls and [0])
def make_F_RK4():
    x = casadi.SX.sym("x", nx)
    u = casadi.SX.sym("u", nu)
    k1 = f_cont(x, u)[0]
    k2 = f_cont(x + dt/2 * k1, u)[0]
    k3 = f_cont(x + dt/2 * k2, u)[0]
    k4 = f_cont(x + dt * k3, u)[0]
    x_next = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return casadi.Function("F_RK4", [x, u], [x_next], ["x", "u"], ["x_next"])

F_RK4 = make_F_RK4()

# cost functions
def compute_stage_cost(x, u):
    return (casadi.dot(Q @ (x-x_ref), (x-x_ref)) + casadi.dot(R @ (u-u_ref), (u-u_ref))) / 2

def compute_terminal_cost(x):
    return casadi.dot(Qf @ (x-x_ref), (x-x_ref)) / 2

# NLP (fixed horizon, terminal equality)
def make_nlp():
    X = [casadi.SX.sym(f"x{k}", nx) for k in range(K+1)]
    U = [casadi.SX.sym(f"u{k}", nu) for k in range(K)]
    G = []
    J = 0
    for k in range(K):
        J += compute_stage_cost(X[k], U[k]) * dt
        G.append(X[k+1] - F_RK4(X[k], U[k])[0])   # <-- positional call + [0]
    # terminal equality
    G.append(X[-1] - x_ref)
    J += compute_terminal_cost(X[-1])
    Z = casadi.vertcat(*X, *U)
    nlp = {"x": Z, "f": J, "g": casadi.vertcat(*G)}
    solver = casadi.nlpsol("solver_fixed", "ipopt", nlp,
                           {'ipopt': {'max_iter':200, 'print_level':0}, 'print_time': False})
    return solver

def compute_optimal_control(Solver, x_init, x0_guess=None):
    x0_val = float(x_init.full().ravel()[0])
    lbx = []; ubx = []
    for _ in range(K+1):
        lbx += x_lb; ubx += x_ub
    for _ in range(K):
        lbx += u_lb; ubx += u_ub
    lbx[0] = ubx[0] = float(min(max(x0_val, x_lb[0]), x_ub[0]))
    total_vars = nx*(K+1) + nu*K
    if x0_guess is None:
        x0 = [0.0]*total_vars; x0[0] = x0_val
    else:
        x0 = list(x0_guess.full().ravel())
        if len(x0) != total_vars:
            x0 = [0.0]*total_vars; x0[0] = x0_val
    neq = nx*K + nx
    lbg = [0.0]*neq; ubg = [0.0]*neq
    res = Solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
    z = res["x"].full().ravel()
    u0 = z[nx*(K+1): nx*(K+1) + nu]
    return casadi.DM(u0), casadi.DM(z)

# Simulation
solver_fixed = make_nlp()
t_final = 60.0
t_eval = np.arange(0.0, t_final, dt)
x_current = casadi.DM([0.2])
X_log = []; U_log = []
x0_guess = casadi.DM.zeros(nx*(K+1) + nu*K)

for t in t_eval:
    u_opt, sol = compute_optimal_control(solver_fixed, x_current, x0_guess)
    hdot = f_cont(x_current, u_opt)[0]
    w = disturbance(t)
    x_next = x_current + dt * (hdot + w / A)
    X_log.append(float(x_next.full().ravel()[0]))
    U_log.append(float(u_opt.full().ravel()[0]))
    x_current = casadi.DM(x_next)
    x0_guess = sol

X_arr = np.array(X_log)[:len(t_eval)]
U_arr = np.array(U_log)[:len(t_eval)]

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t_eval, X_arr, label="h"); plt.axhline(h_ref, linestyle="--", color="gray", label="h_ref")
plt.ylim(0,1.05); plt.legend(); plt.grid(True); plt.title("Fixed-horizon MPC")

plt.subplot(1,2,2)
plt.step(t_eval, U_arr, where='post', label="q_i"); plt.ylim(-0.1,10.1); plt.legend(); plt.grid(True)
plt.title("Control input (fixed)")

plt.tight_layout()
plt.savefig("images/tank_mpc_fixed_fixedcalls.png")
plt.show()
print("Saved images/tank_mpc_fixed_fixedcalls.png")