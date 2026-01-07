import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time   # ★追加：計算時間計測用

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

N_long  = 15
N_short = 5

# ヒステリシス付き閾値（正規化スラック用）
s_high = 5e-3
s_low  = 1e-4

# dwell time
min_dwell = 5

h_min, h_max = 0.0, 1.2
u_min, u_max = 0.0, 3.0

# ==============================
# 外乱
# ==============================
def disturbance(t):
    return -1.0 if 20.0 <= t <= 25.0 else 0.0

# ==============================
# 平滑化関数（変更点①）
# ==============================
def smooth_relu(x, eps=1e-4):
    return 0.5 * (x + ca.sqrt(x**2 + eps))

def make_f():
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w")

    h  = smooth_relu(x[0])
    qi = smooth_relu(u[0] + w)

    h_dot = -(S_out/A) * ca.sqrt(2*g*h + 1e-6) + qi/A
    return ca.Function("f", [x, u, w], [h_dot])

f = make_f()

# ==============================
# 実機用 RK4
# ==============================
def rk4(x, u, w):
    k1 = f(x, u, w)
    k2 = f(x + dt/2*k1, u, w)
    k3 = f(x + dt/2*k2, u, w)
    k4 = f(x + dt*k3, u, w)
    return x + dt*(k1+2*k2+2*k3+k4)/6

# ==============================
# MPC用 RK4（変更点②）
# ==============================
def rk4_sym(x, u):
    k1 = f(x, u, 0)
    k2 = f(x + dt/2*k1, u, 0)
    k3 = f(x + dt/2*k2, u, 0)
    k4 = f(x + dt*k3, u, 0)
    return x + dt*(k1+2*k2+2*k3+k4)/6

# ==============================
# MPCソルバ生成
# ==============================
def make_solver(N):
    X = [ca.SX.sym(f"x{k}", nx) for k in range(N+1)]
    U = [ca.SX.sym(f"u{k}", nu) for k in range(N)]

    J = 0
    G = []

    for k in range(N):
        J += 0.5 * ca.mtimes([(X[k]-x_ref).T, Q, (X[k]-x_ref)])
        J += 0.5 * ca.mtimes([U[k].T, R, U[k]])

        # 変更点③：Δuペナルティ
        if k > 0:
            J += 0.5 * 10.0 * ca.sumsqr(U[k] - U[k-1])

        G.append(X[k+1] - rk4_sym(X[k], U[k]))

    J += 0.5 * ca.mtimes([(X[-1]-x_ref).T, Qf, (X[-1]-x_ref)])

    return ca.nlpsol(
        "solver", "ipopt",
        {"x": ca.vertcat(*X, *U), "f": J, "g": ca.vertcat(*G)},
        {"ipopt.print_level": 0, "print_time": 0}
    )

solver_long  = make_solver(N_long)
solver_short = make_solver(N_short)

# ==============================
# シミュレーション
# ==============================
def simulate(variable=True):
    x = ca.DM([0.8])
    x_pred_prev = x

    mode = "long"
    dwell = 0
    s_bar = 0.0

    X_log, U_log, N_log, S_log, T_log = [], [], [], [], []  # ★T_log追加

    x0 = None

    for t in t_grid:
        w = disturbance(t)

        # ---- スラック定義（1ステップ先予測誤差・正規化）
        s = abs(float(x - x_pred_prev)) / (h_max - h_min)

        alpha = 0.9
        s_bar = alpha * s_bar + (1 - alpha) * s

        # ---- ホライズン切替
        if variable and dwell == 0:
            if mode == "long" and s_bar > s_high:
                mode, dwell = "short", min_dwell
            elif mode == "short" and s_bar < s_low:
                mode, dwell = "long", min_dwell

        dwell = max(dwell - 1, 0)
        if not variable:
            mode = "long"

        solver, N = (solver_short, N_short) if mode == "short" else (solver_long, N_long)



        nxu = nx*(N+1) + nu*N
        lbx = [h_min]*(nx*(N+1)) + [u_min]*(nu*N)
        ubx = [h_max]*(nx*(N+1)) + [u_max]*(nu*N)
        lbx[0] = ubx[0] = float(x)

        if x0 is None or len(x0) != nxu:
            x0 = [float(x)] * nxu

        # ★計算時間計測開始
        t_start = time.perf_counter()

        sol = solver(
            lbx=lbx, ubx=ubx,
            lbg=[0]*(nx*N), ubg=[0]*(nx*N),
            x0=x0
        )

        # ★計算時間計測終了
        T_log.append(time.perf_counter() - t_start)

        solx = sol["x"].full().ravel()

        u = ca.DM([solx[nx*(N+1)]])
        x_pred_prev = ca.DM([solx[nx]])

        x0 = np.concatenate([solx[nx:], solx[-nu:]])

        x = rk4(x, u, w)

        X_log.append(float(x))
        U_log.append(float(u))
        N_log.append(N)
        S_log.append(s_bar)

    return (
        np.array(X_log),
        np.array(U_log),
        np.array(N_log),
        np.array(S_log),
        np.array(T_log)
    )

# ==============================
# 実行
# ==============================
Xf, Uf, Nf, Sf, Tf = simulate(variable=False)
Xv, Uv, Nv, Sv, Tv = simulate(variable=True)

# ==============================
# プロット
# ==============================
fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)

fixed_style = dict(color="tab:blue", linestyle="--", linewidth=2)
var_style   = dict(color="tab:red", linestyle="-", linewidth=2)

axs[0].plot(t_grid, Xf, label="Fixed MPC", **fixed_style)
axs[0].plot(t_grid, Xv, label="Variable MPC", **var_style)
axs[0].axhline(h_ref, color="gray", linestyle=":")
axs[0].set_ylabel("h")
axs[0].set_title("State Response")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_grid, Uf, label="Fixed MPC", **fixed_style)
axs[1].plot(t_grid, Uv, label="Variable MPC", **var_style)
axs[1].set_ylabel("u")
axs[1].set_title("Control Input")
axs[1].legend()
axs[1].grid(True)

axs[2].step(t_grid, Nf, where="post", label="Fixed MPC", **fixed_style)
axs[2].step(t_grid, Nv, where="post", label="Variable MPC", **var_style)
axs[2].set_ylabel("N")
axs[2].set_title("Horizon Switching")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(t_grid, Sf, label="Fixed MPC", **fixed_style)
axs[3].plot(t_grid, Sv, label="Variable MPC", **var_style)
axs[3].set_ylabel("Slack (EMA)")
axs[3].set_title("Prediction Error Indicator")
axs[3].legend()
axs[3].grid(True)

axs[4].plot(t_grid, Tf, label="Fixed MPC", **fixed_style)
axs[4].plot(t_grid, Tv, label="Variable MPC", **var_style)
axs[4].set_ylabel("Time [s]")
axs[4].set_xlabel("Time [s]")
axs[4].set_title("MPC Computation Time per Step")
axs[4].legend()
axs[4].grid(True)

plt.tight_layout()
plt.show()
