import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

# =========================================
# パラメータ
# =========================================
A = 1.0
S_out = 0.1
g = 9.81

nx, nu = 1, 1
h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([50.0])
R  = ca.diag([0.1])
rho_s = 50.0          # スラック重み（重要）

dt = 0.2
t_end = 60
t_grid = np.arange(0, t_end, dt)

# ホライズン候補
N_long, N_mid, N_short = 15, 6, 3

# スラック閾値
s_skip    = 0.005
s_trigger = 0.05

# 制約
h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# =========================================
# 外乱
# =========================================
def disturbance(t):
    return -1.0 if 20 <= t <= 25 else 0.0

# =========================================
# タンクモデル（PDF準拠）
# =========================================
def make_f():
    x = ca.SX.sym("x")
    u = ca.SX.sym("u")
    w = ca.SX.sym("w")

    h = ca.fmax(x, 0)
    qi = ca.fmax(u + w, 0)
    hdot = -(S_out/A)*ca.sqrt(2*g*h) + qi/A
    return ca.Function("f", [x,u,w], [hdot])

f = make_f()

def rk4(x,u,w):
    k1 = f(x,u,w)
    k2 = f(x+dt/2*k1,u,w)
    k3 = f(x+dt/2*k2,u,w)
    k4 = f(x+dt*k3,u,w)
    return ca.fmax(x + dt*(k1+2*k2+2*k3+k4)/6, h_min)

# =========================================
# 終端スラック付き MPC
# =========================================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}") for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}") for k in range(N)]
    s = ca.SX.sym("s")

    J, G = 0, []

    for k in range(N):
        J += (X[k]-x_ref).T@Q@(X[k]-x_ref)/2
        J += U[k].T@R@U[k]/2
        G.append(X[k+1] - X[k] - dt*f(X[k],U[k],0))

    # 終端制約 + スラック
    G.append(X[-1] - (x_ref + s))
    J += (X[-1]-x_ref).T@Qf@(X[-1]-x_ref)/2
    J += rho_s * s**2

    opt = ca.vertcat(*X,*U,s)
    g   = ca.vertcat(*G)

    return ca.nlpsol("solver","ipopt",
        {"x":opt,"f":J,"g":g},
        {"ipopt.print_level":0,"print_time":0}
    )

solvers = {
    N_long: make_solver(N_long),
    N_mid: make_solver(N_mid),
    N_short: make_solver(N_short)
}

# =========================================
# シミュレーション
# =========================================
x = ca.DM([0.8])
u_prev = ca.DM([0.0])
last_s = 0.0

X_log, U_log, N_log, S_log = [], [], [], []

for t in t_grid:

    # --- スラックに基づく切替 ---
    if abs(last_s) < s_skip:
        u = u_prev
        N = 0
        s_val = last_s
    else:
        if abs(last_s) > s_trigger:
            N = N_short
        else:
            N = N_mid

        solver = solvers[N]
        lbx = [h_min]*(N+1) + [u_min]*N + [-1.0]
        ubx = [h_max]*(N+1) + [u_max]*N + [ 1.0]
        lbx[0] = ubx[0] = float(x)

        sol = solver(
            lbx=lbx, ubx=ubx,
            lbg=[0]*(N+1), ubg=[0]*(N+1),
            x0=[float(x)]*(N+1+N+1)
        )

        sol_x = sol["x"].full().ravel()
        u = ca.DM([sol_x[N+1]])
        s_val = sol_x[-1]

    # 状態更新
    x = rk4(x, u, disturbance(t))
    last_s = s_val
    u_prev = u

    X_log.append(float(x))
    U_log.append(float(u))
    N_log.append(N)
    S_log.append(abs(s_val))

# =========================================
# 描画
# =========================================
fig, axs = plt.subplots(4,1,figsize=(10,8),sharex=True)

axs[0].plot(t_grid,X_log)
axs[0].axhline(h_ref,color="gray",linestyle=":")
axs[0].set_ylabel("h")

axs[1].step(t_grid,U_log,where="post")
axs[1].set_ylabel("u")

axs[2].step(t_grid,N_log,where="post")
axs[2].set_ylabel("Horizon N")

axs[3].plot(t_grid,S_log)
axs[3].set_ylabel("|s|")
axs[3].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("images/slack_driven_variable_mpc.png")
plt.show()
