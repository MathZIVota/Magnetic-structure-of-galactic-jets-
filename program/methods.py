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
    if kind in (None, "none", "zero_reg"):
        return None                       # <-- НОВОЕ: без штрафа
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
        self.M = M
        self.H = H
        self.G = M.T @ M
        self.MT = M.T

    def __call__(self, y, alpha):
        if self.H is None:
            # без регуляризации — обычный МНК (псевдообратный)
            # lstsq даёт устойчивое решение даже для плохо обусловленной M
            f, *_ = np.linalg.lstsq(self.M, y, rcond=None)
            return f
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
    if field_2d.get('mode', 'from_A') == 'direct':
        print("⚠️ mode='direct': A_phi не задан, графики A_phi пропущены.")
        return
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
    if field_2d.get('mode', 'from_A') == 'direct':
        print("⚠️ mode='direct': A_phi не задан, графики A_phi пропущены.")
        return
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
    
# ==============================================================================
# 6. СОХРАНЕНИЕ И ЗАГРУЗКА МНОГОЗАПУСКОВЫХ ДАННЫХ
# ==============================================================================

import os
import pandas as pd


def append_run_to_csv(filename, run_name, data, rho):
    """
    Дописывает результаты одного запуска в общий CSV.

    Каждый запуск хранится отдельными столбцами:
        rho | <run_name>__B_z_true | <run_name>__B_z_rec
            | <run_name>__alpha    | <run_name>__delta
            | <run_name>__err_Bz

    Если файл уже есть, новые столбцы добавляются через merge по rho.
    """
    # --- DataFrame текущего запуска ---
    df_new = pd.DataFrame({'rho': rho})
    for key, val in data.items():
        df_new[f"{run_name}__{key}"] = np.asarray(val)

    # --- Если файл есть — merge по rho ---
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        # Если rho совпадает (один и тот же срез) — просто конкат
        if np.allclose(df_old['rho'].values, df_new['rho'].values):
            df_out = pd.concat([df_old, df_new.drop(columns=['rho'])], axis=1)
        else:
            df_out = pd.merge(df_old, df_new, on='rho', how='outer')
    else:
        df_out = df_new

    d = os.path.dirname(filename)
    if d:
        os.makedirs(d, exist_ok=True)
    df_out.to_csv(filename, index=False)
    print(f"💾 Запуск '{run_name}' добавлен в {filename}")


def load_runs(filename):
    """
    Читает CSV со множеством запусков, возвращает:
        rho : 1D массив
        runs : dict {run_name: {'B_z_true': ..., 'B_z_rec': ..., ...}}
    """
    df = pd.read_csv(filename)
    rho = df['rho'].values

    runs = {}
    suffixes = set()
    for col in df.columns:
        if col == 'rho':
            continue
        if '__' not in col:
            continue
        run_name, key = col.split('__', 1)
        suffixes.add(key)

    for run_name in sorted({c.split('__', 1)[0] for c in df.columns if '__' in c}):
        runs[run_name] = {}
        for key in suffixes:
            col = f"{run_name}__{key}"
            if col in df.columns:
                runs[run_name][key] = df[col].values
    return rho, runs

def _normalize(y, mode="max"):
    """
    Нормирует профиль для отображения.

    mode:
      "max"  — делить на max(|y|)         (форма профиля, диапазон [-1, 1])
      "l2"   — делить на ||y||_2          (единичная энергия)
      "peak" — делить на max(y)           (сохраняет знак, не работает при y<=0)
      None   — без нормировки
    """
    y = np.asarray(y, float)
    if mode is None:
        return y
    if mode == "max":
        m = np.max(np.abs(y))
        return y / m if m > 0 else y
    if mode == "l2":
        m = np.linalg.norm(y)
        return y / m if m > 0 else y
    if mode == "peak":
        m = np.max(y)
        return y / m if m != 0 else y
    raise ValueError(f"Неизвестный режим нормировки: {mode}")

def plot_all_true_fields(rho, runs, field_key='B_z_true',
                         title=None, xlabel=r'$\rho$', ylabel=None,
                         norm="max"):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        if field_key in d:
            y = _normalize(d[field_key], mode=norm)
            ax.plot(rho, y, lw=2, color=cmap(i / max(n - 1, 1)), label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel((ylabel or field_key) + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or f"Все истинные поля: {field_key}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_recovered_fields(rho, runs, true_key='B_z_true',
                              rec_key='B_z_rec',
                              title=None, xlabel=r'$\rho$', ylabel=None,
                              norm="max", ax=None, legend=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        color = cmap(i / max(n - 1, 1))
        if true_key in d:
            y_true = _normalize(d[true_key], mode=norm)
            ax.plot(rho, y_true, lw=2, color=color, label=f'{name} (true)')
        if rec_key in d:
            a = float(np.atleast_1d(d.get('alpha', np.nan)).ravel()[0])
            y_rec = _normalize(d[rec_key], mode=norm)
            ax.plot(rho, y_rec, lw=2, ls='--', color=color,
                    label=f'{name} (rec, α={a:.1e})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel((ylabel or rec_key) + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or "Восстановление полей: все запуски")
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7, ncol=1)
    return ax

def plot_all_rm(rho, runs, rm_key='RM_even_true',
                title=None, xlabel=r'$\rho$', ylabel=None,
                norm="max"):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        if rm_key in d:
            y = _normalize(d[rm_key], mode=norm)
            ax.plot(rho, y, lw=2, color=cmap(i / max(n - 1, 1)), label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel((ylabel or rm_key) + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or f"Все RM: {rm_key}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_rm_noisy(rho, runs,
                      rm_noisy_key='RM_even_noisy',
                      rm_true_key='RM_even_true',
                      title=None, xlabel=r'$\rho$', ylabel=None,
                      norm="max", ax=None, legend=True):
    """
    Все зашумлённые RM на одном графике.
    Если ax=None — создаёт свою фигуру. Иначе рисует на переданный ax.
    Истинная кривая нормируется на тот же масштаб, что и шумная,
    чтобы величина шума визуально не терялась.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        color = cmap(i / max(n - 1, 1))
        y_true  = d.get(rm_true_key,  None)
        y_noisy = d.get(rm_noisy_key, None)
        if y_true is None or y_noisy is None:
            continue

        # нормируем шумную на масштаб истинной (чтобы шум не «растягивался»)
        y_true  = np.asarray(y_true,  float)
        y_noisy = np.asarray(y_noisy, float)
        if norm == "max":
            m = np.max(np.abs(y_true))
            if m > 0:
                y_true, y_noisy = y_true / m, y_noisy / m
        elif norm == "l2":
            m = np.linalg.norm(y_true)
            if m > 0:
                y_true, y_noisy = y_true / m, y_noisy / m

        ax.plot(rho, y_true,  lw=1, color=color, alpha=0.4)
        ax.plot(rho, y_noisy, 'o-', ms=2, lw=1.2, color=color,
                label=f'{name} (noisy)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel((ylabel or rm_noisy_key) + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or "Зашумлённые RM: все запуски")
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7)
    return ax
    
# ==============================================================================
# 7. ПОСТРОЕНИЕ В ПОЛЬЗОВАТЕЛЬСКИЙ ax (для subplots-сеток)
# ==============================================================================

def _plot_rm_noisy_into(ax, rho, runs,
                        rm_noisy_key='RM_even_noisy',
                        rm_true_key='RM_even_true',
                        norm="max",
                        title=None):
    """Рисует все RM (истинные фоном + шумные) в переданный ax."""
    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        color = cmap(i / max(n - 1, 1))
        if rm_true_key in d and rm_noisy_key in d:
            # общий масштаб нормировки — от истинного
            m = None
            if norm == "max":
                m = np.max(np.abs(d[rm_true_key]))
            elif norm == "l2":
                m = np.linalg.norm(d[rm_true_key])
            if m and m > 0:
                y_true  = d[rm_true_key]  / m
                y_noisy = d[rm_noisy_key] / m
            else:
                y_true, y_noisy = d[rm_true_key], d[rm_noisy_key]
            ax.plot(rho, y_true, lw=1, color=color, alpha=0.35)
            ax.plot(rho, y_noisy, 'o-', ms=2, lw=1.2, color=color,
                    label=f'{name} (noisy)')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(rm_noisy_key + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or "Зашумлённые RM: все запуски")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _plot_recovered_into(ax, rho, runs,
                         true_key='B_z_true', rec_key='B_z_rec',
                         norm="max", title=None):
    """Рисует все восстановленные B_z (пунктир) + истинные (сплошные) в переданный ax."""
    cmap = plt.get_cmap('viridis')
    n = len(runs)
    for i, (name, d) in enumerate(runs.items()):
        color = cmap(i / max(n - 1, 1))
        if true_key in d:
            ax.plot(rho, _normalize(d[true_key], norm), lw=1.8,
                    color=color, label=f'{name} (true)')
        if rec_key in d:
            a = float(np.atleast_1d(d.get('alpha', np.nan)).ravel()[0])
            ax.plot(rho, _normalize(d[rec_key], norm), '--', lw=1.8,
                    color=color, label=f'{name} (rec, α={a:.1e})')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(rec_key + (f" (norm={norm})" if norm else ""))
    ax.set_title(title or "Восстановление полей: все запуски")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
# ==============================================================================
# 8. СВОДНАЯ СЕТКА ПО ВСЕМ δ
# ==============================================================================

def plot_delta_grid(deltas, filename_pattern,
                    norm="max",
                    rm_noisy_key='RM_even_noisy',
                    rm_true_key='RM_even_true',
                    rec_key='B_z_rec',
                    true_key='B_z_true',
                    figsize_per_row=(14, 4),
                    runs_filter=None):
    """
    Строит сетку 2*len(deltas) x 1 subplots.

    Для каждого delta:
      - строка 2k   : plot_all_rm_noisy     (зашумлённые RM для всех запусков)
      - строка 2k+1 : plot_all_recovered_fields (восстановление B_z для всех запусков)

    Параметры
    ----------
    deltas : список float
        Уровни шума.
    filename_pattern : str
        Шаблон имени файла. Должен содержать "{delta}", например:
            "results/all_runs_var_with B_noise_{delta}.csv"
        Функция подставит значение и попытается открыть файл.
        Дополнительно пробует несколько форматов: "0.02", "0.020", "0.0".
    norm : str | None
        Нормировка графиков ("max", "l2", None).
    runs_filter : callable | None
        Функция (name, d) -> bool. Позволяет показать подмножество запусков.
    """
    n_delta = len(deltas)
    if n_delta == 0:
        print("deltas пуст.")
        return

    rows = 2 * n_delta
    fig_h = figsize_per_row[1] * rows
    fig, axes = plt.subplots(rows, 1, figsize=(figsize_per_row[0], fig_h))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for k, delta_tmp in enumerate(deltas):
        # --- ищем файл по шаблону ---
        candidates = [
            filename_pattern.format(delta=delta_tmp),
            filename_pattern.format(delta=f"{delta_tmp:.3f}"),
            filename_pattern.format(delta=f"{delta_tmp:.2f}"),
            filename_pattern.format(delta=f"{delta_tmp:g}"),
        ]
        fname = next((c for c in candidates if os.path.exists(c)), None)

        ax_rm   = axes[2 * k]
        ax_rec  = axes[2 * k + 1]

        if fname is None:
            ax_rm.set_title(f"δ = {delta_tmp} — файл не найден")
            ax_rec.set_title(f"δ = {delta_tmp} — файл не найден")
            ax_rm.axis("off"); ax_rec.axis("off")
            continue

        rho, runs = load_runs(fname)

        if runs_filter is not None:
            runs = {n: d for n, d in runs.items() if runs_filter(n, d)}

        _plot_rm_noisy_into(
            ax_rm, rho, runs,
            rm_noisy_key=rm_noisy_key,
            rm_true_key=rm_true_key,
            norm=norm,
            title=f"δ = {delta_tmp} — зашумлённые RM",
        )
        _plot_recovered_into(
            ax_rec, rho, runs,
            true_key=true_key,
            rec_key=rec_key,
            norm=norm,
            title=f"δ = {delta_tmp} — восстановление B_z",
        )

    plt.tight_layout()
    plt.show()
    
def plot_noise_sweep_grid(csv_paths, deltas,
                          rm_noisy_key='RM_even_noisy',
                          rm_true_key='RM_even_true',
                          rec_true_key='B_z_true',
                          rec_key='B_z_rec',
                          norm="max",
                          figsize_per_row=(12, 4),
                          title=None):
    """
    Сводная сетка: строки = уровни шума δ, столбцы = [RM noisy | восстановление].

    Параметры
    ----------
    csv_paths : dict {delta: filename}
        Пути к CSV, по одному на каждый уровень шума.
    deltas : list
        Порядок уровней шума по строкам (может быть отсортирован).
    norm : str | None
        Нормировка внутри каждой функции (см. _normalize).
    """
    n_rows = len(deltas)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        squeeze=False,
    )

    for row, d in enumerate(deltas):
        fname = csv_paths[d]
        if not os.path.exists(fname):
            print(f"⚠️ Файл не найден: {fname}")
            continue

        rho, runs = load_runs(fname)

        # --- левая колонка: зашумлённые RM ---
        plot_all_rm_noisy(
            rho, runs,
            rm_noisy_key=rm_noisy_key,
            rm_true_key=rm_true_key,
            norm=norm,
            ax=axes[row, 0],
            legend=(row == 0),           # легенда только на первой строке
        )
        axes[row, 0].set_title(f"RM даже при δ={d}", fontsize=11)

        # --- правая колонка: восстановление B_z ---
        plot_all_recovered_fields(
            rho, runs,
            true_key=rec_true_key,
            rec_key=rec_key,
            norm=norm,
            ax=axes[row, 1],
            legend=(row == 0),
        )
        axes[row, 1].set_title(f"Восстановление B_z при δ={d}", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        plt.tight_layout()
    plt.show()
    
# ==============================================================================
# 9. ГРАФИКИ ОШИБОК ВОССТАНОВЛЕНИЯ
# ==============================================================================

def plot_all_errors_bar(rho, runs,
                        err_key='err_Bz',
                        title=None, ax=None, sort_by_error=True):
    """
    Столбчатая диаграмма относительной ошибки err_Bz для всех запусков
    (при фиксированном уровне шума δ).

    runs : dict {run_name: {'err_Bz': массив, ...}}
           Ошибка берётся как среднее по rho (или единственное значение).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    # собираем средние ошибки
    data = []
    for name, d in runs.items():
        if err_key not in d:
            continue
        arr = np.atleast_1d(np.asarray(d[err_key], float))
        data.append((name, float(np.mean(arr))))

    if sort_by_error:
        data.sort(key=lambda x: x[1])

    names  = [x[0] for x in data]
    values = [x[1] for x in data]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]

    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(r'Относительная ошибка $\varepsilon_{B_z}$')
    ax.set_title(title or "Средняя относительная ошибка восстановления $B_z$")
    ax.grid(axis='y', alpha=0.3)

    # подписи значений над столбцами
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7)

    return ax


def plot_error_vs_noise(csv_paths, deltas,
                        err_key='err_Bz',
                        title=None, ax=None,
                        loglog=True, annotate_runs=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    # --- собираем ошибки ---
    err_by_run = {}
    for d in deltas:
        fname = csv_paths[d]
        if not os.path.exists(fname):
            print(f"⚠️ Файл не найден: {fname}")
            continue
        _, runs = load_runs(fname)
        for name, data in runs.items():
            if err_key not in data:
                continue
            arr = np.atleast_1d(np.asarray(data[err_key], float))
            err_by_run.setdefault(name, {})[d] = float(np.mean(arr))

    if not err_by_run:
        print("Нет данных для построения графика.")
        return ax

    cmap = plt.get_cmap('viridis')
    names = sorted(err_by_run.keys())
    n = len(names)

    # -----------------------------------------------------------------
    # вспомогательная функция: нарисовать линию ε = δ в нужном масштабе
    # -----------------------------------------------------------------
    def _draw_eps_eq_delta(color='k', ls='--', lw=1, alpha=0.6, label=r'$\varepsilon = \delta$'):
        pos_deltas = [d for d in deltas if d > 0] or [d for d in deltas]
        if not pos_deltas:
            return
        xmin, xmax = min(pos_deltas), max(pos_deltas)
        if xmin == xmax:
            xmin, xmax = xmin * 0.5, xmax * 2 or 1.0
        xx = np.linspace(xmin, xmax, 50) if not loglog \
             else np.logspace(np.log10(xmin), np.log10(xmax), 50)
        ax.plot(xx, xx, color=color, ls=ls, lw=lw, alpha=alpha, label=label)

    # -----------------------------------------------------------------
    # ветка: по одной кривой на каждый run_name
    # -----------------------------------------------------------------
    if annotate_runs:
        for i, name in enumerate(names):
            ds = sorted(err_by_run[name].keys())
            es = [err_by_run[name][d] for d in ds]
            color = cmap(i / max(n - 1, 1))

            if loglog:
                ds_plot = [max(x, 1e-6) for x in ds]
                es_plot = [max(x, 1e-6) for x in es]
                ax.loglog(ds_plot, es_plot, 'o-', color=color, lw=1.8,
                          ms=6, label=name)
            else:
                ax.plot(ds, es, 'o-', color=color, lw=1.8, ms=6, label=name)

        _draw_eps_eq_delta()

    # -----------------------------------------------------------------
    # ветка: усреднение по всем запускам
    # -----------------------------------------------------------------
    else:
        ds, es_mean, es_std = [], [], []
        for d in sorted(deltas):
            vals = [err_by_run[name][d]
                    for name in err_by_run if d in err_by_run[name]]
            if not vals:
                continue
            ds.append(d)
            es_mean.append(float(np.mean(vals)))
            es_std.append(float(np.std(vals)))

        if loglog:
            ds_plot = [max(x, 1e-6) for x in ds]
            es_plot = [max(x, 1e-6) for x in es_mean]
            ax.errorbar(ds_plot, es_plot, yerr=es_std, fmt='o-',
                        color='tab:blue', lw=2, ms=7, capsize=4,
                        label='Среднее ± σ')
        else:
            ax.errorbar(ds, es_mean, yerr=es_std, fmt='o-',
                        color='tab:blue', lw=2, ms=7, capsize=4,
                        label='Среднее ± σ')

        _draw_eps_eq_delta()

    ax.set_xlabel(r'Уровень шума $\delta$')
    ax.set_ylabel(r'Относительная ошибка $\varepsilon_{B_z}$')
    ax.set_title(title or r"Зависимость ошибки восстановления от уровня шума")
    ax.grid(True, which='both' if loglog else 'major', ls=':', alpha=0.5)
    ax.legend(fontsize=8)
    return ax
