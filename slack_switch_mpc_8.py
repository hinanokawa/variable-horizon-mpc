import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# =========================
# パラメータ
# =========================
A = 1.0
S_out = 0.1
g = 9.81

dt = 0.2
T = 60.0
t_grid = np.arange(0, T, dt)

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([50.0])
R  = ca.diag([0.1])

h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# ホライズン候補
N_long  = 15
N_mid   = 6
N_short = 3

# スラック判定閾値
s_skip    = 0.01
s_trigger = 0.05

lambda_s = 50.0   # 終端スラック重み

# =========================
# 外乱（急変）
# =========================
def disturbance(t):
    if 20 <= t <= 25:
        return -1.0
    return 0.0

# =========================
# タンクモデル
# =========================
def make_f():
    x = ca.SX.sym("x",1)
    u = ca.SX.sym("u",1)
    w = ca.SX.sym("w")

    h = ca.fmax(x[0],0)
    qi = ca.fmax(u[0] + w,0)

    hdot = -(S_out/A)*ca.sqrt(2*g*h) + qi/A
    return ca.Function("f",[x,u,w],[hdot])

f = make_f()

def rk4(x,u,w):
    k1 = f(x,u,w)
    k2 = f(x+dt/2*k1,u,w)
    k3 = f(x+dt/2*k2,u,w)
    k4 = f(x+dt*k3,u,w)
    return ca.fmax(x + dt*(k1+2*k2+2*k3+k4)/6, h_min)

# =========================
# MPC (終端スラック付き)
# =========================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}",1) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}",1) for k in range(N)]
    s = ca.SX.sym("s",1)

    J = 0
    G = []

    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T,Q,(X[k]-x_ref)])/2
        J += ca.mtimes([U[k].T,R,U[k]])/2
        G.append(X[k+1]-X[k]-dt*f(X[k],U[k],0))

    # 終端スラック
    G.append(X[-1] - x_ref - s)
    J += ca.mtimes([s.T, lambda_s, s])/2
    J += ca.mtimes([(X[-1]-x_ref).T,Qf,(X[-1]-x_ref)])/2

    opt = ca.vertcat(*X,*U,s)
    g = ca.vertcat(*G)

    return ca.nlpsol("solver","ipopt",
        {"x":opt,"f":J,"g":g},
        {"ipopt.print_level":0,"print_time":0})

# ソルバ生成
solvers = {
    N_long:  make_solver(N_long),
    N_mid:   make_solver(N_mid),
    N_short: make_solver(N_short)
}

# =========================
# シミュレーション
# =========================
def simulate(variable=True):
    x = ca.DM([0.8])
    u_prev = ca.DM([0.0])

    last_s = np.inf   # 初回は必ず解く

    Xlog,Ulog,Nlog,Slog,solve_log = [],[],[],[],[]

    for t in t_grid:

        # --- 判定 ---
        solve = True
        if variable and abs(last_s) < s_skip:
            solve = False

        if solve:
            if not variable:
                N = N_mid
            else:
                if abs(last_s) > s_trigger:
                    N = N_short
                else:
                    N = N_long

            solver = solvers[N]

            lbx = [h_min]*(N+1) + [u_min]*N + [-1e2]
            ubx = [h_max]*(N+1) + [u_max]*N + [1e2]
            lbx[0] = ubx[0] = float(x)

            sol = solver(
                lbx=lbx, ubx=ubx,
                lbg=[0]*(N+1), ubg=[0]*(N+1),
                x0=[float(x)]*(len(lbx))
            )

            z = sol["x"].full().ravel()
            u = ca.DM([z[(N+1)]])
            last_s = z[-1]
            solved = 1
        else:
            u = u_prev
            N = 0
            solved = 0

        w = disturbance(t)
        x = rk4(x,u,w)

        Xlog.append(float(x))
        Ulog.append(float(u))
        Nlog.append(N)
        Slog.append(abs(last_s))
        solve_log.append(solved)

        u_prev = u

    return map(np.array,(Xlog,Ulog,Nlog,Slog,solve_log))

# 実行
Xf,Uf,Nf,Sf,Solvef = simulate(variable=False)
Xv,Uv,Nv,Sv,Solvev = simulate(variable=True)

# =========================
# 描画
# =========================
fig,axs = plt.subplots(4,1,figsize=(10,9),sharex=True)

axs[0].plot(t_grid,Xf,'--',label="Fixed MPC")
axs[0].plot(t_grid,Xv,label="Variable MPC")
axs[0].axhline(h_ref,color='gray',linestyle=':')
axs[0].set_ylabel("h")
axs[0].legend()

axs[1].step(t_grid,Uf,'--',where='post',label="Fixed")
axs[1].step(t_grid,Uv,where='post',label="Variable")
axs[1].set_ylabel("u")
axs[1].legend()

axs[2].step(t_grid,Nv,where='post')
axs[2].set_ylabel("Horizon N")

axs[3].plot(t_grid,Sv)
axs[3].set_ylabel("|terminal slack|")
axs[3].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()
