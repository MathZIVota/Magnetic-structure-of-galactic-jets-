"""
Magnet_Field.py
===============
Физика магнитного поля из векторного потенциала.

Соглашение: A = (0, A_phi, 0) в цилиндрических координатах (rho, phi, z).

Ротор в цилиндрических координатах для A = A_phi(rho, z) * e_phi:
    B_rho = - dA_phi / dz
    B_phi =   0
    B_z   = (1/rho) * d(rho * A_phi) / drho

То есть продольное поле B_z и радиальное B_rho определяются A_phi,
а B_phi тождественно равно нулю (это важно: в исходной постановке
B_phi бралось из RM_odd, но при A = A_phi*e_phi оно равно нулю;
см. примечание в конце файла).
"""

import numpy as np
import pandas as pd
import os


# ==============================================================================
# 1. ЗАДАНИЕ ВЕКТОРНОГО ПОТЕНЦИАЛА
# ==============================================================================

def make_A_phi(func, **params):
    """
    Оборачивает пользовательскую функцию A_phi(rho, z, **params)
    в замыкание с фиксированными параметрами.
    """
    def A_phi(rho, z):
        return func(rho, z, **params)
    return A_phi


# --- Примеры профилей A_phi (можно добавлять свои) ---------------------------

def A_phi_gaussian(rho, z, A0=1.0, rho_scale=5.0, z_scale=3.0):
    """A_phi = A0 * rho * exp(-(rho/rho_scale)^2 - (z/z_scale)^2)."""
    return A0 * rho * np.exp(-(rho / rho_scale) ** 2 - (z / z_scale) ** 2)


def A_phi_uniform(rho, z, A0=1.0):
    """A_phi = A0 * rho  ->  однородное B_z = 2*A0."""
    return A0 * rho


def A_phi_power(rho, z, A0=1.0, power=2.0, rho_scale=5.0):
    """A_phi = A0 * rho * (rho/rho_scale)^power * exp(-rho/rho_scale)."""
    return A0 * rho * (rho / rho_scale) ** power * np.exp(-rho / rho_scale)


def A_phi_screen(rho, z, A0=1.0, rho_scale=5.0, z_scale=3.0):
    """A_phi = A0 * rho * exp(-rho/rho_scale) * cos(z/z_scale)."""
    return A0 * rho * np.exp(-rho / rho_scale) * np.cos(z / z_scale)


# ==============================================================================
# 2. РАСЧЁТ B = rot(A) НА 2D-СЕТКЕ
# ==============================================================================

def compute_B_from_A_phi(A_phi_func, rho_vals, z_vals, B_phi_func=None):
    """
    Считает компоненты B на сетке (rho, z) из заданного A_phi(rho, z).

    Возвращает словарь с 2D-массивами:
        'rho', 'z'  : 1D сетки
        'A_phi'     : 2D массив A_phi
        'B_rho'     : -dA_phi/dz
        'B_phi'     : 0 (тождественно)
        'B_z'       : (1/rho) d(rho*A_phi)/drho
        'B_mag'     : модуль B
    """
    rho_vals = np.asarray(rho_vals, float)
    z_vals   = np.asarray(z_vals,   float)
    RHO, Z   = np.meshgrid(rho_vals, z_vals)          # shape (Nz, Nrho)

    A_phi_2d = np.asarray(A_phi_func(RHO, Z), float)

    drho = rho_vals[1] - rho_vals[0] if len(rho_vals) > 1 else 1.0
    dz   = z_vals[1]   - z_vals[0]   if len(z_vals)   > 1 else 1.0

    # B_rho = - dA_phi / dz
    B_rho_2d = -np.gradient(A_phi_2d, dz, axis=0)

    # B_z = (1/rho) * d(rho * A_phi) / drho
    rho_A_phi = RHO * A_phi_2d
    d_rho_A   = np.gradient(rho_A_phi, drho, axis=1)

    RHO_safe = np.where(np.abs(RHO) < 1e-12, 1e-12, RHO)
    B_z_2d   = d_rho_A / RHO_safe

    # На оси симметрии: B_z(rho->0) = 2 * dA_phi/drho|_{rho=0}
    # Берём крайнее левое значение как экстраполяцию
    B_z_2d[:, 0] = 2.0 * (A_phi_2d[:, 1] - A_phi_2d[:, 0]) / drho
    B_z_2d[-1, :] = B_z_2d[-2, :]
    
    if B_phi_func is not None:
        B_phi_2d = np.asarray(B_phi_func(RHO, Z), float)
    else:
        B_phi_2d = np.zeros_like(A_phi_2d)
        
    B_mag_2d = np.sqrt(B_rho_2d**2 + B_z_2d**2)

    return {
        'rho': rho_vals, 'z': z_vals,
        'A_phi': A_phi_2d,
        'B_rho': B_rho_2d,
        'B_phi': B_phi_2d,
        'B_z':   B_z_2d,
        'B_mag': B_mag_2d,
        'mode':  'from_A',
    }
    



# ==============================================================================
# 3. ОСЕВОЙ ПРОФИЛЬ (для 1D обратной задачи)
# ==============================================================================

def extract_axial_profiles(field, z_index=None):
    """
    Возвращает 1D-профили по rho при фиксированном z (по умолчанию — центр).
    Используется как "истина" для обратной задачи RM.
    """
    Nz = len(field['z'])
    if z_index is None:
        z_index = Nz // 2

    return {
        'z_value':  field['z'][z_index],
        'rho':      field['rho'],
        'A_phi':    field['A_phi'][z_index, :],
        'B_rho':    field['B_rho'][z_index, :],
        'B_z':      field['B_z'][z_index, :],
        'B_phi':    field['B_phi'][z_index, :],
        'B_mag':    field['B_mag'][z_index, :],
    }


# ==============================================================================
# 4. СОХРАНЕНИЕ / ЗАГРУЗКА
# ==============================================================================

def save_field_to_csv(field, filename):
    """Сохраняет 2D-поле (все компоненты) в длинный CSV."""
    RHO, Z = np.meshgrid(field['rho'], field['z'])
    df = pd.DataFrame({
        'rho':   RHO.ravel(),
        'z':     Z.ravel(),
        'A_phi': field['A_phi'].ravel(),
        'B_rho': field['B_rho'].ravel(),
        'B_phi': field['B_phi'].ravel(),
        'B_z':   field['B_z'].ravel(),
        'B_mag': field['B_mag'].ravel(),
    })
    d = os.path.dirname(filename)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Поле сохранено: {filename}")


def load_field_from_csv(filename):
    """Загружает поле из CSV, возвращает словарь той же структуры."""
    df = pd.read_csv(filename)
    rho = np.sort(df['rho'].unique())
    z   = np.sort(df['z'].unique())
    Nz, Nrho = len(z), len(rho)

    def _pivot(col):
        return df.pivot(index='z', columns='rho', values=col).values

    return {
        'rho': rho, 'z': z,
        'A_phi': _pivot('A_phi'),
        'B_rho': _pivot('B_rho'),
        'B_phi': _pivot('B_phi'),
        'B_z':   _pivot('B_z'),
        'B_mag': _pivot('B_mag'),
    }
    
# ==============================================================================
# 5. ПРЯМОЕ ЗАДАНИЕ ПОЛЯ (без векторного потенциала)
# ==============================================================================

def make_B_profile(func, **params):
    """
    Оборачивает пользовательскую функцию B_z(rho, **params)
    в замыкание с фиксированными параметрами.
    Аналогично make_A_phi, но для компоненты B.
    """
    def B_z(rho):
        return func(rho, **params)
    return B_z


def compute_field_from_B(rho_vals, B_z_func, B_rho_func=None,
                         B_phi_func=None, z_vals=None):
    rho = np.asarray(rho_vals, float)
    N_rho = len(rho)

    B_z_1d   = np.asarray(B_z_func(rho), float) if B_z_func is not None else np.zeros(N_rho)
    B_rho_1d = np.asarray(B_rho_func(rho), float) if B_rho_func is not None else np.zeros(N_rho)

    if z_vals is None:
        z_vals = np.array([0.0])
    z_vals = np.asarray(z_vals, float)
    Nz = len(z_vals)

    RHO, Z = np.meshgrid(rho, z_vals)
    
    # Вычисляем B_phi на 2D сетке
    if B_phi_func is not None:
        B_phi_2d = np.asarray(B_phi_func(RHO, Z), float)
    else:
        B_phi_2d = np.zeros((Nz, N_rho))

    # Размножаем 1D профили на 2D сетку
    B_z_2d   = np.tile(B_z_1d, (Nz, 1))
    B_rho_2d = np.tile(B_rho_1d, (Nz, 1))
    B_mag_2d = np.sqrt(B_rho_2d**2 + B_phi_2d**2 + B_z_2d**2)

    return {
        'rho': rho, 'z': z_vals, 'A_phi': np.zeros((Nz, N_rho)),
        'B_rho': B_rho_2d, 'B_phi': B_phi_2d, 'B_z': B_z_2d, 'B_mag': B_mag_2d,
        'mode': 'direct',
    }
