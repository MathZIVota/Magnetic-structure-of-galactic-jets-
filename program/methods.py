"""
methods.py
==========
Ядро: операторы RM, регуляризация, обратная задача, метрики, графики.
Блокнот импортирует отсюда готовые функции и не содержит их реализаций.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# ==============================================================================
# 1. ОПЕРАТОРЫ ПРЯМОЙ ЗАДАЧИ RM
# ==============================================================================

def _interpolation_matrix(r_q, rho):
    """Линейная интерполяция с сетки rho на точки r_q."""
    W = np.zeros((len(r_q), len(rho)))
    for i, rq in enumerate(r_q):
        idx = np.searchsorted(rho, rq)
        if idx == 0:
            W[i, 0] = 1.0
        elif idx == len(rho):
            W[i, -1] = 1.0
        else:
            w1 = (rho[idx] - rq) / (rho[idx] - rho[idx-1])
            w2 = (rq - rho[idx-1]) / (rho[idx] - rho[idx-1])
            W[i, idx-1] = w1
            W[i, idx]   = w2
    return W


def build_operators(rho, K_RM=1.0, theta_deg=80.0, n_quad=400):
    """
    Матрицы M_z (вклад B_z в RM_even) и M_phi (вклад B_phi в RM_odd).

    Сингулярность 1/sqrt(r^2 - x^2) устраняется заменой r = sqrt(x^2 + u^2).
    """
    rho = np.asarray(rho, float)
    N   = len(rho)
    R   = rho[-1]
    cot_theta = 1.0 / np.tan(np.deg2rad(theta_deg))

    M_z   = np.zeros((N, N))
    M_phi = np.zeros((N, N))

    t = np.linspace(0.0, 1.0, n_quad)
    w = np.full(n_quad, 1.0 / (n_quad - 1))
    w[0] *= 0.5
    w[-1] *= 0.5

    for i, x in enumerate(rho):
        T_max = np.sqrt(max(R**2 - x**2, 0.0))
        if T_max < 1e-12:
            continue
        r_q = np.sqrt(x**2 + (T_max * t)**2)
        W   = _interpolation_matrix(r_q, rho)

        M_z[i] = 2.0 * K_RM * cot_theta * T_max * (w @ W)

        kernel_factor = np.where(r_q > 1e-12, x / (r_q**2), 0.0)
        M_phi[i] = 2.0 * K_RM * T_max * ((w * kernel_factor) @ W)

    return M_z, M_phi


def forward_rm(M_z, M_phi, f_z, f_phi):
    """Прямая задача: RM_even = M_z f_z, RM_odd = M_phi f_phi, RM_total = сумма."""
    even = M_z @ f_z
    odd  = M_phi @ f_phi
    return even, odd, even + odd


# ==============================================================================
# 2. РЕГУЛЯРИЗАЦИЯ
# ==============================================================================

def regularizer(N, kind="second_order"):
    """Матрица штрафа H (penalty = f^T H f)."""
    if kind == "zero_order":
        return np.eye(N)
    if kind == "first_order":
        D1 = np.diff(np.eye(N), axis=0)
        return D1.T @ D1
    if kind == "second_order":
        D1 = np.diff(np.eye(N), axis=0)
        D2 = np.diff(D1, axis=0)
        return D2.T @ D2
    raise ValueError(f"Неизвестный регуляризатор: {kind}")


class RegularizedInversion:
    """Кэширует G = M^T M и M^T для быстрого перебора alpha."""
    def __init__(self, M, H):
        self.M  = M
        self.H  = H
        self.G  = M.T @ M
        self.MT = M.T

    def __call__(self, y, alpha):
        return solve(self.G + alpha * self.H, self.MT @ y, assume_a="pos")


# ==============================================================================
# 3. ВЫБОР alpha ПО L-КРИВОЙ
# ==============================================================================

def choose_alpha_lcurve(inv, y, alphas, label="B", ax=None):
    """Строит L-кривую, возвращает alpha в 'углу' (нормированное расстояние до хорды)."""
    residuals, penalties = [], []
    for a in alphas:
        f = inv(y, a)
        residuals.append(np.linalg.norm(inv.M @ f - y))
        penalties.append(np.sqrt(max(f @ inv.H @ f, 0.0)))

    residuals = np.maximum(np.asarray(residuals), 1e-30)
    penalties = np.maximum(np.asarray(penalties), 1e-30)

    lx = np.log(residuals); ly = np.log(penalties)
    lx = (lx - lx.min()) / max(np.ptp(lx), 1e-12)
    ly = (ly - ly.min()) / max(np.ptp(ly), 1e-12)

    p1 = np.array([lx[0],  ly[0]])
    p2 = np.array([lx[-1], ly[-1]])
    chord = p2 - p1
    nc = np.linalg.norm(chord)

    if nc < 1e-12:
        best = 0
    else:
        dist = [abs(chord[0]*(p1[1]-y_) - chord[1]*(p1[0]-x_)) / nc
                for x_, y_ in zip(lx, ly)]
        best = int(np.argmax(dist))

    if ax is not None:
        ax.loglog(penalties, residuals,  "o-", ms=4)
        ax.scatter(penalties[best], residuals[best],  s=90,
                   facecolors="none", edgecolors="red", lw=2)
        ax.set_xlabel("Штраф √(fᵀHf)")
        ax.set_ylabel("Невязка ‖Mf − y‖")
        ax.set_title(f"L-кривая: {label}\nα = {alphas[best]:.2e}")

    return float(alphas[best]), residuals, penalties


# ==============================================================================
# 4. МЕТРИКИ И ШУМ
# ==============================================================================

def electron_density(rho, model="constant", scale=None):
    """Профиль концентрации электронов n_e(rho)."""
    rho = np.asarray(rho, float)
    if scale is None:
        scale = rho[-1] / 3.0 if len(rho) else 1.0
    if model == "constant":
        return np.ones_like(rho)
    if model == "gaussian":
        return np.exp(-(rho / scale) ** 2)
    if model == "power_law":
        safe = np.where(rho == 0, 1e-9, rho)
        return (safe / scale) ** -1.0
    raise ValueError(f"Неизвестный n_e_model: {model}")


def rel_error(true, approx):
    """Относительная L2-ошибка."""
    nt = np.linalg.norm(true)
    if nt == 0:
        return 0.0
    return np.linalg.norm(approx - true) / nt


relative_error = rel_error  # алиас


def add_noise(y, delta, scale=None, seed=None):
    """Аддитивный гауссов шум с амплитудой delta * max|y|."""
    y = np.asarray(y, float)
    if delta == 0:
        return y.copy()
    if scale is None or scale == 0:
        scale = float(np.max(np.abs(y))) or 1.0
    rng = np.random.default_rng(seed)
    return y + delta * scale * rng.standard_normal(y.shape)


# ==============================================================================
# 5. ГРАФИКИ
# ==============================================================================

def plot_recovery(rho, true_fields, rec_fields, alphas, delta, title="", trivial_phi=False):
    """2x2: истина vs восстановленное + относительные ошибки."""
    B_phi_true, B_z_true = true_fields
    B_phi_rec,  B_z_rec  = rec_fields
    a_phi, a_z = alphas

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(rho, B_phi_true, 'k-', lw=2, label='Истинное')
    if not trivial_phi:
        ax.plot(rho, B_phi_rec, 'r--', lw=2, label=f'Восстановл. (α={a_phi:.1e})')
    ax.set_title(r"$B_\phi$"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if not trivial_phi:
        floor = 0.02 * max(np.max(np.abs(B_phi_true)), 1e-12)
        rel = np.abs(B_phi_rec - B_phi_true) / np.maximum(np.abs(B_phi_true), floor)
        ax.semilogy(rho, rel, 'r-', lw=2)
    ax.set_title(r"Отн. ошибка $B_\phi$"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rho, B_z_true, 'k-', lw=2, label='Истинное')
    ax.plot(rho, B_z_rec,  'b--', lw=2, label=f'Восстановл. (α={a_z:.1e})')
    ax.set_title(r"$B_z$"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    floor = 0.02 * max(np.max(np.abs(B_z_true)), 1e-12)
    rel = np.abs(B_z_rec - B_z_true) / np.maximum(np.abs(B_z_true), floor)
    ax.semilogy(rho, rel, 'b-', lw=2)
    ax.set_title(r"Отн. ошибка $B_z$"); ax.grid(alpha=0.3)

    fig.suptitle(f"Восстановление полей (шум δ={delta})", fontsize=15, fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_sweep(df, x_col, y_cols, labels, title, tol=0.1):
    """Лог-лог ошибка vs уровень шума."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col].to_numpy()
    colors = ["tab:blue", "tab:red", "tab:green"]
    for i, (col, lab) in enumerate(zip(y_cols, labels)):
        y = df[col].to_numpy()
        ok = (x > 0) & np.isfinite(y) & (y > 0)
        if np.any(ok):
            ax.loglog(x[ok], y[ok], "o-", color=colors[i % len(colors)], label=lab)
    if np.any(x > 0):
        xx = np.linspace(x[x > 0].min(), x.max(), 50)
        ax.loglog(xx, xx, "k--", lw=1, alpha=0.6, label="ε = δ")
    ax.axhline(tol, color="gray", ls=":", label=f"допуск {tol:.0%}")
    ax.set_xlabel("Уровень шума δ"); ax.set_ylabel("Отн. ошибка ε")
    ax.set_title(title); ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout(); plt.show()


def plot_A_and_B(field_2d, rho, z):
    """Диагностика: A_phi, B_z, B_rho на 2D-сетке."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [rho[0], rho[-1], z[0], z[-1]]
    for ax, key, title in zip(
        axes,
        ['A_phi', 'B_z', 'B_rho'],
        [r'$A_\phi$', r'$B_z$', r'$B_\rho$']
    ):
        im = ax.imshow(field_2d[key], extent=extent, origin='lower',
                       aspect='auto', cmap='RdBu_r')
        ax.set_xlabel(r'$\rho$'); ax.set_ylabel(r'$z$'); ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.show()

def plot_A_phi_profiles(field_2d, rho, z, z_slices=None, rho_slices=None):
    """
    Строит 1D-зависимости A_phi от rho (при фиксированных z)
    и от z (при фиксированных rho).

    Параметры
    ----------
    field_2d : dict
        Результат compute_B_from_A_phi — должен содержать 'A_phi' (2D массив).
    rho, z : 1D массивы
        Сетки.
    z_slices : список индексов по z (None -> берём 3 характерных: край, центр, край)
    rho_slices : список индексов по rho (None -> 3 характерных)
    """
    A_phi = field_2d['A_phi']
    Nz, Nrho = A_phi.shape

    if z_slices is None:
        z_slices = [0, Nz // 2, Nz - 1]
    if rho_slices is None:
        rho_slices = [0, Nrho // 2, Nrho - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- A_phi от rho при разных z ---
    ax = axes[0]
    for iz in z_slices:
        ax.plot(rho, A_phi[iz, :], lw=2, label=f'z = {z[iz]:.2f}')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$A_\phi(\rho, z)$')
    ax.set_title(r'$A_\phi$ как функция $\rho$ (разные $z$)')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- A_phi от z при разных rho ---
    ax = axes[1]
    for ir in rho_slices:
        ax.plot(z, A_phi[:, ir], lw=2, label=fr'$\rho$ = {rho[ir]:.2f}')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$A_\phi(\rho, z)$')
    ax.set_title(r'$A_\phi$ как функция $z$ (разные $\rho$)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_B_profiles(field_2d, rho, z, z_slices=None, rho_slices=None):
    Nz, Nrho = field_2d['B_z'].shape
    if z_slices is None:
        z_slices = [0, Nz // 2, Nz - 1]
    if rho_slices is None:
        rho_slices = [0, Nrho // 2, Nrho - 1]

    components = [
        ('B_rho', r'$B_\rho$', 'tab:blue'),
        ('B_phi', r'$B_\phi$', 'tab:green'),
        ('B_z',   r'$B_z$',    'tab:red'),
        ('B_mag', r'$|B|$',    'tab:purple'),
    ]

    # Наборы стилей для срезов (различимы и в ч/б)
    line_styles = ['-', '--', ':', '-.']

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    for i, (key, label, base_color) in enumerate(components):
        data = field_2d[key]
        cmap = plt.get_cmap('viridis')

        # --- от rho при разных z ---
        ax = axes[i, 0]
        n = len(z_slices)
        for k, iz in enumerate(z_slices):
            shade = cmap(k / max(n - 1, 1)) if n > 1 else base_color
            ax.plot(rho, data[iz, :], lw=2,
                    color=shade, ls=line_styles[k % len(line_styles)],
                    label=f'z = {z[iz]:.2f}')
        ax.set_xlabel(r'$\rho$'); ax.set_ylabel(label)
        ax.set_title(f'{label} от $\\rho$ (разные $z$)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # --- от z при разных rho ---
        ax = axes[i, 1]
        n = len(rho_slices)
        for k, ir in enumerate(rho_slices):
            shade = cmap(k / max(n - 1, 1)) if n > 1 else base_color
            ax.plot(z, data[:, ir], lw=2,
                    color=shade, ls=line_styles[k % len(line_styles)],
                    label=fr'$\rho$ = {rho[ir]:.2f}')
        ax.set_xlabel(r'$z$'); ax.set_ylabel(label)
        ax.set_title(f'{label} от $z$ (разные $\\rho$)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Компоненты магнитного поля: 1D-профили',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    