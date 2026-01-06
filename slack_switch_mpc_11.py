import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ==============================
# 基本設定
# ==============================
A = 1.0
S_out = 0.1
g = 9.81

nx, nu = 1, 1
dt = 0.2
t_end = 60.0
t_grid = np.arange(0, t_end, dt)

h_ref = 0.8
x_ref = ca.DM([h_ref])

Q  = ca.diag([10.0])
Qf = ca.diag([100.0])
R  = ca.diag([0.1])

# ホライズン
N_long  = 15
N_short = 5

# スラック判定閾値（予測誤差）
s_small = 1e-4
s_large = 5e-3

# 制約
h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# ==============================
# 外乱（急変）
# ==============================
def disturbance(t):
    if 20.0 <= t <= 25.0:
        return -1.0
    return 0.0

# ==============================
# タンクモデル
# ==============================
def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h = ca.fmax(x[0], 0.0)
    qi = ca.fmax(u[0] + w, 0.0)

    h_dot = -(S_out/A)*ca.sqrt(2*g*h) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

def rk4(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return ca.fmax(x + dt*(k1+2*k2+2*k3+k4)/6, h_min)

# ==============================
# MPC ソルバ（固定ホライズン）
# ==============================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]

    J = 0
    G = []

    for k in range(N):
        J += ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])/2
        J += ca.mtimes([U[k].T, R, U[k]])/2
        G.append(X[k+1] - X[k] - dt*f(X[k], U[k], 0))

    J += ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])/2

    opt = ca.vertcat(*X, *U)
    g   = ca.vertcat(*G)

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": opt, "f": J, "g": g},
        {"ipopt.print_level": 0, "print_time": 0}
    )
    return solver

solver_long  = make_solver(N_long)
solver_short = make_solver(N_short)

# ==============================
# シミュレーション
# ==============================
def simulate(variable=True):
    x = ca.DM([0.8])
    u_prev = ca.DM([0.4])

    # 前回 MPC の 1-step 予測
    x_pred_prev = x

    X_log, U_log, N_log, S_log = [], [], [], []

    for t in t_grid:
        w = disturbance(t)

        # ==========================
        # スラック（予測誤差）
        # ==========================
        s = abs(float(x - x_pred_prev))

        # ==========================
        # ホライズン・計算切替
        # ==========================
        if variable:
            if s > s_large:
                solve = True
                N = N_short      # 急変 → 即応
            elif s < s_small:
                solve = True
                N = N_long       # 安定 → 長期計画
            else:
                solve = False    # 中間 → 再計算スキップ
                N = 0
        else:
            solve = True
            N = N_long

        # ==========================
        # MPC 計算
        # ==========================
        if solve:
            solver = solver_short if N == N_short else solver_long
            nxu = nx*(N+1) + nu*N

            lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
            ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
            lbx[0] = ubx[0] = float(x)

            sol = solver(
                lbx=lbx, ubx=ubx,
                lbg=[0]*(nx*N),
                ubg=[0]*(nx*N),
                x0=[float(x)]*nxu
            )

            solx = sol["x"].full().ravel()
            u = ca.DM([solx[nx*(N+1)]])
            x_pred_prev = ca.DM([solx[nx]])  # 1-step 予測
        else:
            u = u_prev

        # ==========================
        # システム更新
        # ==========================
        x = rk4(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(s)

        u_prev = u

    return np.array(X_log), np.array(U_log), np.array(N_log), np.array(S_log)

# ==============================
# 実行
# ==============================
Xf, Uf, Nf, Sf = simulate(variable=False)
Xv, Uv, Nv, Sv = simulate(variable=True)

# ==============================
# 描画
# ==============================
fig, axs = plt.subplots(4,1, figsize=(10,8), sharex=True)

axs[0].plot(t_grid, Xf, "--", label="Fixed MPC")
axs[0].plot(t_grid, Xv, label="Variable MPC")
axs[0].axhline(h_ref, color="gray", linestyle=":")
axs[0].set_ylabel("h")
axs[0].legend()

axs[1].step(t_grid, Uf, "--", where="post", label="Fixed")
axs[1].step(t_grid, Uv, where="post", label="Variable")
axs[1].set_ylabel("u")
axs[1].legend()

axs[2].step(t_grid, Nv, where="post")
axs[2].set_ylabel("Horizon N")

axs[3].plot(t_grid, Sv)
axs[3].set_ylabel("Slack |x - x̂|")
axs[3].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()
