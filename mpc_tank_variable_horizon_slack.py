# mpc_tank_variable_horizon_slack_fixed.py
import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

# ===== Tank parameters =====
g = 9.81
A = 1.0
C = 0.5

nx = 1
nu = 1

Q  = ca.DM([100.0])
Qf = ca.DM([200.0])
R  = ca.DM([0.1])

# horizon candidates
K_small = 10
K_mid   = 20
K_large = 30

# NOTE: dt is fixed here (based on K_mid). You can instead set dt = T/K if you want fixed total horizon time.
T = 2.0
dt = T / K_mid

h_min = 0.0
h_max = 2.0
u_min = 0.0
u_max = 1.5

h_ref_val = 1.0
x_ref = ca.DM([h_ref_val])
u_ref = ca.DM([0.5])

#######################
# dynamic model
#######################
def make_f():
    states = ca.SX.sym('states', nx)
    ctrls  = ca.SX.sym('ctrls', nu)

    h = states[0]
    qin = ctrls[0]

    qout = C * ca.sqrt(2*g*ca.fmax(h, 0))
    dhdt = (1.0/A) * (qin - qout)
    return ca.Function("f", [states, ctrls], [ca.vertcat(dhdt)])

def make_F_RK4(dt):
    states = ca.SX.sym('states', nx)
    ctrls  = ca.SX.sym('ctrls', nu)
    f = make_f()

    r1 = f(states, ctrls)
    r2 = f(states + dt*r1/2, ctrls)
    r3 = f(states + dt*r2/2, ctrls)
    r4 = f(states + dt*r3, ctrls)
    next_state = states + dt*(r1 + 2*r2 + 2*r3 + r4)/6
    return ca.Function("F_RK4", [states, ctrls], [next_state])

def stage_cost(x,u):
    xd = x - x_ref
    ud = u - u_ref
    return 0.5*(ca.dot(Q@xd, xd) + ca.dot(R@ud, ud))

def terminal_cost(x):
    xd = x - x_ref
    return 0.5*ca.dot(Qf@xd, xd)

#######################
# Build NLP for given K
#######################
def make_nlp(K):
    F = make_F_RK4(dt)

    X = [ca.SX.sym(f"x_{k}", nx) for k in range(K+1)]
    U = [ca.SX.sym(f"u_{k}", nu) for k in range(K)]

    J = 0
    G = []

    for k in range(K):
        J += stage_cost(X[k], U[k]) * dt
        # F returns a 1x1 SX; ensure shapes align
        G.append(X[k+1] - F(X[k], U[k]))

    J += terminal_cost(X[-1])

    vars = ca.vertcat(*X, *U)
    g = ca.vertcat(*G)

    nlp = {"x": vars, "f": J, "g": g}
    opts = {"print_time": False, "ipopt": {"max_iter": 80, "print_level": 0}}
    S = ca.nlpsol("S", "ipopt", nlp, opts)
    return S, X, U

#######################
# integrator
#######################
def make_integrator(dt):
    states = ca.SX.sym("states", nx)
    ctrls  = ca.SX.sym("ctrls", nu)
    f = make_f()
    ode = f(states, ctrls)
    dae = {"x": states, "p": ctrls, "ode": ode}
    return ca.integrator("I", "cvodes", dae, {"tf": dt})

###########################
# create or resize initial guess for given K
###########################
def make_initial_guess(x_now, K):
    total = nx*(K+1) + nu*K
    x0 = ca.DM.zeros(total)
    # states
    for k in range(K+1):
        start = k*nx
        end = start + nx
        x0[start:end] = x_now
    # inputs
    offset = nx*(K+1)
    for k in range(K):
        x0[offset + k*nu : offset + (k+1)*nu] = u_ref
    return x0

###########################
# solve MPC for given K
###########################
def compute_MPC(S, x_now, x0_guess, K):
    # x_now: casadi.DM of size nx
    total = nx*(K+1) + nu*K

    lbx, ubx = [], []
    for k in range(K+1):
        if k == 0:
            lbx += list(x_now.full().ravel())
            ubx += list(x_now.full().ravel())
        else:
            lbx += [h_min]
            ubx += [h_max]

    for _ in range(K):
        lbx += [u_min]
        ubx += [u_max]

    lbg = [0.0] * (nx*K)
    ubg = [0.0] * (nx*K)

    # ensure x0_guess has correct size
    if x0_guess is None or x0_guess.numel() != total:
        x0_guess = make_initial_guess(x_now, K)

    # call solver
    res = S(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    sol = res["x"]
    offset = nx*(K+1)
    u_opt = sol[offset: offset + nu]
    return ca.DM(u_opt), sol

###########################
# simulation loop
###########################
I = make_integrator(dt)
x_now = ca.DM([0.2])
x0_guess = None

X_hist = []
U_hist = []
K_hist = []

prev_x = x_now
prev_u = u_ref

t_sim = np.arange(0, 30, dt)

for t in t_sim:
    # compute slack (simple rule: state change + scaled input change)
    slack = float(abs(x_now - prev_x)) + 0.5 * float(abs(prev_u - u_ref))

    # select horizon
    if slack < 0.02:
        K = K_large
    elif slack < 0.05:
        K = K_mid
    else:
        K = K_small

    # Important: rebuild solver & initial guess for this K
    S, X_sym, U_sym = make_nlp(K)
    # build/resize x0_guess to match this K (warm-start optional)
    if x0_guess is None or x0_guess.numel() != (nx*(K+1) + nu*K):
        x0_guess = make_initial_guess(x_now, K)

    # solve
    u_opt, sol = compute_MPC(S, x_now, x0_guess, K)

    # store the solution as new initial guess for *same K* next time (warm-start)
    x0_guess = sol

    # step system with integrator (expects p shaped like control)
    # ensure p has correct shape (nu,)
    p = ca.DM(u_opt)
    resI = I(x0=x_now, p=p)
    x_next = resI["xf"]

    X_hist.append(float(x_now))
    U_hist.append(float(u_opt))
    K_hist.append(K)

    prev_x = x_now
    prev_u = u_opt
    x_now = x_next

# append last state
X_hist.append(float(x_now))

# convert to numpy
X = np.array(X_hist)
U = np.array(U_hist)
K_hist = np.array(K_hist)

# Plot results + horizon history
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(np.arange(len(X))*dt, X, label="h")
plt.axhline(h_ref_val, linestyle="--", color="k", label="h_ref")
plt.xlabel("Time [s]"); plt.ylabel("Water level [m]"); plt.legend()

plt.subplot(1,3,2)
plt.step(np.arange(len(U))*dt, U, where="post")
plt.xlabel("Time [s]"); plt.ylabel("Inflow [m^3/s]")

plt.subplot(1,3,3)
plt.step(np.arange(len(K_hist))*dt, K_hist, where="post")
plt.xlabel("Time [s]"); plt.ylabel("Horizon K")
plt.ylim(0, K_large + 5)

plt.tight_layout()
plt.savefig("images/tank_variable_horizon_results.png")
plt.show()
print("Saved images/tank_variable_horizon_results.png")
