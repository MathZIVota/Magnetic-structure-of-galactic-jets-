import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from Magnetic_Field import (
    make_A_phi, A_phi_gaussian, A_phi_uniform, A_phi_power, A_phi_screen,
    compute_B_from_A_phi, extract_axial_profiles,
    save_field_to_csv, load_field_from_csv,
    compute_field_from_B
)
from methods import (
    build_operators, forward_rm, regularizer, RegularizedInversion,
    choose_alpha_lcurve, electron_density, rel_error, relative_error,
    add_noise, plot_recovery, plot_sweep, plot_A_and_B,
    plot_A_phi_profiles, plot_B_profiles, 
    append_run_to_csv, load_runs,
    plot_all_true_fields, plot_all_recovered_fields,
    plot_all_rm, plot_all_rm_noisy,
)


def Calculate(A_phi_func=None, B_z_func=None, B_rho_func=None, B_phi_func=None,
              mode="from_A",                       # "from_A" | "direct"
              delta=0.05, reg_kind="second_order",
              n_e_model="constant",
              plot=True, filename=None, run_name=None, seed=42,
              wrap_with_A0=True):

    # ==========================================================================
    # Сетка
    # ==========================================================================
    N_rho, N_z = 200, 120
    RHO_MAX, Z_MAX = 15.0, 8.0
    rho_vals = np.linspace(0.05, RHO_MAX, N_rho)
    z_vals   = np.linspace(-Z_MAX, Z_MAX, N_z)

    # ==========================================================================
    # Поле: два режима
    # ==========================================================================
    if mode == "from_A":
        if A_phi_func is None:
            raise ValueError("mode='from_A' требует A_phi_func")
        if wrap_with_A0:
            A_phi_func = make_A_phi(A_phi_func, A0=1.0)
        field = compute_B_from_A_phi(A_phi_func, rho_vals, z_vals)

    elif mode == "direct":
        if B_z_func is None:
            raise ValueError("mode='direct' требует B_z_func")
        field = compute_field_from_B(
            rho_vals,
            B_z_func=B_z_func,
            B_rho_func=B_rho_func,
            B_phi_func=B_phi_func,
        )
    else:
        raise ValueError(f"Неизвестный mode={mode}")

    # ==========================================================================
    # Осевой срез — единый для обоих режимов
    # ==========================================================================
    prof = extract_axial_profiles(field)
    rho        = prof['rho']
    B_z_true   = prof['B_z']
    B_phi_true = prof['B_phi']

    # ==========================================================================
    # Дальше всё как было: физика, RM, шум, восстановление, сохранение
    # ==========================================================================
    K_RM, theta_deg = 1.0, 80.0
    n_e_true = electron_density(rho, model=n_e_model, scale=RHO_MAX/3.0)

    M_z, M_phi = build_operators(rho, K_RM=K_RM, theta_deg=theta_deg, n_quad=400)

    f_z_true   = n_e_true * B_z_true
    f_phi_true = n_e_true * B_phi_true

    RM_even_true, RM_odd_true, RM_total_true = forward_rm(
        M_z, M_phi, f_z_true, f_phi_true)

    RM_even_noisy = add_noise(RM_even_true, delta, seed=seed)
    RM_odd_noisy  = add_noise(RM_odd_true,  delta, seed=seed + 1)

    N = len(rho)
    if reg_kind in (None, "none"):
        # без регуляризации — alpha не нужен
        H_z   = None
        H_phi = None
        inv_z   = RegularizedInversion(M_z,   None)
        inv_phi = RegularizedInversion(M_phi, None)
        alpha_z_opt   = 0.0
        alpha_phi_opt = 0.0
    else:
        H_z   = regularizer(N, kind=reg_kind)
        H_phi = regularizer(N, kind=reg_kind)
        inv_z   = RegularizedInversion(M_z,   H_z)
        inv_phi = RegularizedInversion(M_phi, H_phi)
        alphas = np.logspace(-10, 0, 80)
        alpha_z_opt,   _, _ = choose_alpha_lcurve(inv_z,   RM_even_noisy, alphas, label="B_z")
        alpha_phi_opt, _, _ = choose_alpha_lcurve(inv_phi, RM_odd_noisy,  alphas, label="B_phi")

    f_z_rec   = inv_z(RM_even_noisy, alpha_z_opt)
    f_phi_rec = inv_phi(RM_odd_noisy, alpha_phi_opt)

    n_e_safe  = np.where(n_e_true > 1e-10, n_e_true, 1e-10)
    B_z_rec   = f_z_rec   / n_e_safe
    B_phi_rec = f_phi_rec / n_e_safe

    err_B_z   = relative_error(B_z_true,   B_z_rec)
    err_B_phi = relative_error(B_phi_true, B_phi_rec)

    # ==========================================================================
    # Сохранение
    # ==========================================================================
    if filename is not None:
        if run_name is None:
            base = (A_phi_func.__name__ if mode == "from_A"
                    else getattr(B_z_func, '__name__', 'B_direct'))
            run_name = f"{mode}_{base}_d{delta:.3f}"
        append_run_to_csv(filename, run_name, {
            'B_z_true':      B_z_true,
            'B_z_rec':       B_z_rec,
            'B_phi_true':    B_phi_true,
            'B_phi_rec':     B_phi_rec,
            'RM_even_true':  RM_even_true,
            'RM_even_noisy': RM_even_noisy,
            'RM_odd_true':   RM_odd_true,
            'RM_odd_noisy':  RM_odd_noisy,
            'alpha':         np.full_like(rho, alpha_z_opt),
            'delta':         np.full_like(rho, delta),
            'err_Bz':        np.full_like(rho, err_B_z),
        }, rho)

    # ==========================================================================
    # Графики
    # ==========================================================================
    if plot:
        if mode == "from_A":
            plot_A_and_B(field, rho_vals, z_vals)
            plot_A_phi_profiles(field, rho_vals, z_vals)
        plot_B_profiles(field, rho_vals, z_vals)

        plot_recovery(
            rho=rho,
            true_fields=(B_phi_true, B_z_true),
            rec_fields=(B_phi_rec, B_z_rec),
            alphas=(alpha_phi_opt, alpha_z_opt),
            delta=delta,
            trivial_phi=True,
        )

    return {
        'rho': rho,
        'B_z_true': B_z_true, 'B_z_rec': B_z_rec,
        'B_phi_true': B_phi_true, 'B_phi_rec': B_phi_rec,
        'RM_even_true': RM_even_true, 'RM_even_noisy': RM_even_noisy,
        'RM_odd_true': RM_odd_true, 'RM_odd_noisy': RM_odd_noisy,
        'alpha': alpha_z_opt, 'delta': delta,
        'err_Bz': err_B_z, 'err_B_phi': err_B_phi,
        'mode': mode,
    }