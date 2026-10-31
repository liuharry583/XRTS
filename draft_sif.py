"""Measured-SIF version of draft: reuse its physics; replace its Gaussian response.

No call to draft.compute() and no convolution of an already-broadened total.
The imported PhysicsConfig.INSTRUMENT_FWHM_eV is unused in this version.
All returned spectra retain draft's angular-frequency spectral-density units
before normalization, although the convolution quadrature uses energy[eV].
"""
from functools import partial

import numpy as np
import physical_core as physics
from sif_response import MeasuredSIF

PhysicsConfig = physics.PhysicsConfig


def compute_components(theta_deg, E_lo, E_hi, nw=600, method=None, *,
                       config=None, sif=None, step=None):
    """Return raw free, observed free, observed ion, total, all unnormalized.

    E is draft's descending photon shift (redshift negative). The source extends
    over the complete measured response support. near-zero FDT handling and
    all electronic/ionic weights follow draft's existing functions.
    """
    state = physics.refresh_state() if config is None else physics.build_state(config)
    cfg = state.config
    if (not np.isfinite([E_lo, E_hi]).all() or E_lo >= E_hi
            or isinstance(nw, (bool, np.bool_)) or not isinstance(nw, (int, np.integer))
            or nw < 3):
        raise ValueError("Invalid output energy grid")
    sif = MeasuredSIF.load() if sif is None else sif
    k = physics.k_of_theta(theta_deg, config=cfg)
    E = np.linspace(E_lo, E_hi, nw)[::-1].copy()
    omega = -E*physics.e/physics.hbar
    keep = np.abs(omega) > 1e-3*state.wp
    E, omega = E[keep], omega[keep]
    if len(E) < 3:
        raise ValueError("Too few output points after the zero-frequency exclusion")
    selected = physics.resolve_electron_method(k, state, method)
    nu = physics.collision_frequency(k, state)
    if selected == "maxwell":
        evaluate = partial(physics.S_free_mw, k, T=state.T, nu=nu, ne=cfg.ne)
    else:
        evaluate = partial(physics.S_free, k, T=state.T, mu=state.mu, nu=nu, ne=cfg.ne)
    raw = np.array([evaluate(w) for w in omega])
    output_step = (E_hi-E_lo)/(nw-1)
    if step is not None and (not np.isfinite(step) or step <= 0):
        raise ValueError("Source step must be finite and positive")
    target_step = min(output_step, np.median(np.diff(sif.energy)),
                      0.01*state.wp*physics.hbar/physics.e,
                      step if step is not None else output_step)
    # Integer subdivisions keep source/output grid points aligned.
    dx = output_step/max(1, int(np.ceil(output_step/target_step)))
    lo = min(E_lo, E_lo-sif.energy[-1])-2*dx
    hi = max(E_hi, E_hi-sif.energy[0])+2*dx
    grid = E_lo+np.arange(int(np.floor((lo-E_lo)/dx)),
                         int(np.ceil((hi-E_lo)/dx))+1)*dx
    # _sample_source accepts angular frequencies in any order.
    source = physics._sample_source(evaluate, -grid*physics.e/physics.hbar, state.wp)
    observed, valid = sif.convolve(grid, source, E)
    # Reuse the original ion weight without constructing a finite-width peak:
    # at omega=0 and sigma=1, G(0)=1/sqrt(2*pi), hence divide this factor out.
    if cfg.USE_CHIHARA_ION:
        weight = float(physics.S_ion_chihara(
            np.array([0.0]), k, state.T, state.mu, cfg.ne, cfg.ZA, cfg.Zf,
            width=1.0, sii_mode=cfg.SII_MODE)[0]*np.sqrt(2*np.pi))
    else:
        weight = float(physics.f_I_k(k, cfg.ZA)**2)
    # R_E dE = R_omega domega; eV-to-omega Jacobian is essential for ion/free ratio.
    ion = weight*sif.evaluate(E)*physics.hbar/physics.e
    total = cfg.Zf*observed+ion
    for value in (raw, observed, ion, total):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite physical spectrum")
    return dict(E_eV=E, omega=omega, S_free_raw=raw, S_free_observed=observed,
                S_elastic_observed=ion, S_tot_observed=total, k=float(k),
                rms_valid_mask=valid, source_step_eV=float(dx),
                source_range_eV=[float(grid[0]), float(grid[-1])], ion_weight=weight,
                electron_method=selected, nu_rad_s=float(nu))


def compute(theta_deg, E_lo, E_hi, nw=600, method=None, *, config=None,
            normalize=True, sif=None, step=None):
    """Same five outputs as draft.compute: free is RAW, total uses measured SIF.

    normalize=True keeps the original independent own-peak conventions.
    Use compute_components() to get the broadened free component as well.
    """
    result = compute_components(theta_deg, E_lo, E_hi, nw, method,
                                config=config, sif=sif, step=step)
    free, total = result["S_free_raw"], result["S_tot_observed"]
    if normalize:
        free, total = map(physics.normalize_spectrum, (free, total))
    return result["E_eV"], result["omega"], free, total, result["k"]
