# Model switch validation

`draft3_before.py` is the exact pre-edit snapshot. `validate_models.py` compares all compute return arrays byte for byte and checks the migrated functions against the reference ASTs. Results are in `results.json`. `physics_core.py` is unchanged (SHA256 checked).

Run from the project directory with a Python environment providing NumPy >= 2, SciPy, and Matplotlib:

```powershell
python .validation/validate_models.py
```

The validation environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and Matplotlib 3.11.2. Local validation-only dependencies are in `deps/`.

All default regressions have max relative difference 0 and byte-identical return arrays. All four model combinations execute without exceptions or nonfinite values.

**The positive BMA frequency requirement fails in the supplied reference implementation itself.** At the requested conditions and representative frequency zero, both reference and migrated implementations return -4.816228245504897e15 rad/s, or -0.055107398386199255 wp. No sign change, absolute value, or clipping was applied, to honor the faithful transplant constraint. Both BMA spectra have minimum S_tot = -1.8162940701292953e-17. Their peak ratios are arithmetic diagnostics only; they do not describe a valid physical Compton peak.

Peak-height ratio is defined as max(S_ion) / max(Zf*S_free) on the actual 600-point grid, with ELASTIC_WIDTH_eV=18 and E0_keV=4.75. The JSON also records total-spectrum center-to-Compton ratios and peak energies.

The two reference functions already use SI: k/q in m^-1, w and nu in rad/s, T in K, ne in m^-3, mu in J, and mass in kg. The external 10 eV input is converted once to 116045.18121550081 K. The reference's ZA*amu ion-mass approximation and ni-based rs are retained.

The batch validation sets NU_MODE and SII_MODE externally on the imported module. When changing thermodynamic inputs externally, it recomputes the original derived module quantities T, wp, kap, nu, and mu, as required by the existing compute interface.
