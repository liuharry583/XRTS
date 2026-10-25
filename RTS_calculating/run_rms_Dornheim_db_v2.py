"""Fit digitized Dornheim data using stored db_v2 S_tot spectra.

Run from any directory with a NumPy >=2, SciPy, Matplotlib environment.
The default reproduces run_rms_Dornheim's numerical postprocessing,
including its convolution indexing. It scans actual database cases, not
an interpolated 45x45 parameter grid. Database E_eV already uses redshift <0.
No physics_core import, engine recomputation, or spectrum mirroring occurs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve

HERE = Path(__file__).resolve().parent


def clean_data(data):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError('Expected finite two-column x/y data')
    x, y = data[:, 0], data[:, 1].copy()
    y[y < 0] = 0
    unique_x = np.unique(x)
    return unique_x, np.array([np.mean(y[x == v]) for v in unique_x])


def normalize(y, x, label):
    area = np.trapezoid(y, x)
    if not np.isfinite(y).all() or not np.isfinite(area) or area <= 0:
        raise ValueError(f'{label}: nonfinite data or nonpositive area')
    return y / area


def process_spectrum(energy, spectrum, conv_grid, sif, fit_energy, step):
    # Sort the paired arrays: stored compute() sorts omega, hence E descends.
    energy, spectrum = np.asarray(energy), np.asarray(spectrum)
    if energy.ndim != 1 or energy.shape != spectrum.shape:
        raise ValueError('Invalid database spectrum shape')
    if not np.isfinite(energy).all() or not np.isfinite(spectrum).all():
        raise ValueError('Nonfinite database spectrum')
    order = np.argsort(energy)
    energy, spectrum = energy[order], spectrum[order]
    if not np.all(np.diff(energy) > 0):
        raise ValueError('Duplicate database energy coordinates')
    if energy[0] > conv_grid[0] or energy[-1] < conv_grid[-1]:
        raise ValueError('Database energy range does not cover convolution grid')
    raw = interp1d(energy, spectrum, kind='linear', bounds_error=False,
                   fill_value=0)(conv_grid)
    broadened = convolve(raw, sif, mode='same') * step
    fit = interp1d(conv_grid, broadened, kind='linear')(fit_energy)
    return normalize(fit, fit_energy, 'convolved theory')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-dir', type=Path, default=HERE.parent/'database/db_v2')
    parser.add_argument('--exp', type=Path, default=HERE/'exp_data.txt')
    parser.add_argument('--sif', type=Path, default=HERE/'sif_data.txt')
    parser.add_argument('--output-dir', type=Path, default=HERE/'rms_db_v2_results')
    parser.add_argument('--theta', type=float, default=40.0)
    parser.add_argument('--e-lo', type=float, default=-35.0)
    parser.add_argument('--e-hi', type=float, default=-10.0)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--pad', type=float, default=20.0)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    if args.e_lo >= args.e_hi or args.step <= 0 or args.pad < 0:
        raise ValueError('Invalid energy window, step, or pad')
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ex, ey = clean_data(np.loadtxt(args.exp))
    sx, sy = clean_data(np.loadtxt(args.sif))
    mask = (ex >= args.e_lo) & (ex <= args.e_hi)
    fit_x = ex[mask]
    if len(fit_x) < 3:
        raise ValueError('Need at least three experimental fit points')
    exp_norm = normalize(ey[mask], fit_x, 'experiment')
    grid = np.arange(args.e_lo-args.pad, args.e_hi+args.pad, args.step)
    sif_grid = interp1d(sx, sy, kind='linear', bounds_error=False, fill_value=0)(grid)
    sif_grid = normalize(sif_grid, grid, 'SIF')
    with (args.db_dir/'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['status'] in ('done', 'ok') and r['filename']
                and float(r['rho_gcc']) > 0 and float(r['Te_eV']) > 0
                and np.isclose(float(r['theta_deg']), args.theta)]
    if not rows:
        raise ValueError(f'No calculated cases for theta={args.theta}')
    records, spectra = [], {}
    fixed = None
    for row in rows:
        path = args.db_dir/row['filename']
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['meta_json']))
            for key in ('rho_gcc', 'Te_eV', 'theta_deg'):
                if not np.isclose(float(meta[key]), float(row[key])):
                    raise ValueError(f'{path.name}: manifest/metadata mismatch: {key}')
            config = {k: meta.get(k) for k in
                      ('E0_keV', 'ZA', 'Zf', 'material', 'NU_FRAC', 'NU_MODE',
                       'SII_MODE', 'ELASTIC_WIDTH_eV', 'USE_CHIHARA_ION', 'method')}
            if fixed is None:
                fixed = config
            elif fixed != config:
                raise ValueError(f'{path.name}: mixed physics configuration')
            if not np.isclose(float(meta['ZA']), 4) or meta['material'] != 'Be':
                raise ValueError('Expected Be database for this comparison')
            if float(d['nu']) != 0:
                raise ValueError(f'{path.name}: collisions are not OFF')
            y = process_spectrum(d['E_eV'], d['S_tot'], grid, sif_grid, fit_x, args.step)
            ne = float(d['n_e'])
            if not np.isfinite(ne) or ne <= 0:
                raise ValueError(f'{path.name}: invalid electron density')
        te = float(row['Te_eV'])
        rms = float(np.sqrt(np.mean((y-exp_norm)**2)))
        rec = dict(rho_gcc=float(row['rho_gcc']), Te_eV=te, ne_m3=ne,
                   ne_1e23_cm3=ne/1e29, RMS=rms, filename=row['filename'])
        records.append(rec)
        if (ne, te) in spectra:
            raise ValueError('Duplicate density/temperature case')
        spectra[ne, te] = y
    temps = np.array(sorted({r['Te_eV'] for r in records}))
    densities = np.array(sorted({r['ne_m3'] for r in records}))
    if len(temps) < 2 or len(densities) < 2:
        raise ValueError('Need a two-dimensional parameter grid for contours')
    if len(records) != len(temps)*len(densities):
        raise ValueError('Incomplete database grid; refusing to invent missing spectra')
    matrix = np.full((len(densities), len(temps)), np.nan)
    for r in records:
        matrix[np.searchsorted(densities, r['ne_m3']),
               np.searchsorted(temps, r['Te_eV'])] = r['RMS']
    if not np.isfinite(matrix).all():
        raise ValueError('Incomplete/nonfinite RMS matrix')
    best = min(records, key=lambda r:r['RMS'])
    best_y = spectra[best['ne_m3'], best['Te_eV']]
    warnings = []
    if not np.isclose(float(fixed['E0_keV']), 2.96):
        warnings.append(f"Database E0={fixed['E0_keV']} keV differs from Dornheim E0=2.96 keV; this is a workflow comparison, not a matched-condition reproduction.")
    if best['Te_eV'] in (temps[0], temps[-1]) or best['ne_m3'] in (densities[0], densities[-1]):
        warnings.append('Minimum lies on the database grid boundary; no extrapolation performed.')
    warnings.append('Convolution uses the reference script indexing verbatim on its asymmetric energy grid; physical zero-lag centering is not corrected in this compatibility implementation.')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output/'rms_scan.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader()
        writer.writerows(sorted(records, key=lambda r:(r['ne_m3'],r['Te_eV'])))
    np.savez(output/'rms_grid.npz', Te_eV=temps, ne_m3=densities, RMS=matrix)
    np.savetxt(output/'best_fit.txt', np.column_stack((fit_x, exp_norm, best_y)),
               header='energy_shift_eV experiment_normalized theory_convolved_normalized')
    summary = dict(database=str(args.db_dir.resolve()), experimental_data=str(args.exp.resolve()),
                   sif_data=str(args.sif.resolve()), theta_deg=args.theta,
                   fit_window_eV=[args.e_lo,args.e_hi], dE=args.step, pad_eV=args.pad,
                   experimental_points=len(fit_x), cases=len(records),
                   grid_shape=list(matrix.shape), database_physics=fixed,
                   best=best, warnings=warnings, spectrum_field='S_tot',
                   energy_convention='database E_eV used directly; redshift negative',
                   parameter_grid='native database; no parameter interpolation')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    tx, ny = np.meshgrid(temps, densities/1e29)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    filled=ax1.contourf(tx,ny,matrix,levels=40,cmap='viridis')
    fig.colorbar(filled,ax=ax1,label='RMS Error')
    cs=ax1.contour(tx,ny,matrix,levels=15,colors='k',linewidths=0.8,alpha=0.6)
    ax1.clabel(cs,inline=True,fontsize=8,fmt='%.4f')
    label=f"Te={best['Te_eV']:.1f} eV, ne={best['ne_1e23_cm3']:.3f}e23 cm^-3"
    ax1.plot(best['Te_eV'],best['ne_1e23_cm3'],'r*',markersize=15,label='Min RMS'+chr(10)+label)
    ax1.set(xlabel='Electron Temperature (eV)',ylabel=r'Electron Density ($10^{23}$ cm$^{-3}$)',title='RMS Contour (db_v2 native grid)')
    ax1.legend(fontsize=8)
    ax2.plot(fit_x,exp_norm,'g-',lw=2,label='Experiment (Normalized)')
    ax2.plot(fit_x,best_y,'r--',lw=2,label='Best Fit (Convolved)'+chr(10)+label)
    ax2.set(xlabel='Energy Shift (eV)',ylabel='Normalized Intensity',title='Best Fit vs Experiment (Compton Red Wing)')
    ax2.legend(fontsize=8);ax2.grid(True,alpha=0.3)
    fig.suptitle(f"db_v2 E0={fixed['E0_keV']:g} keV; experiment E0=2.96 keV; theta={args.theta:g} deg")
    fig.tight_layout();fig.savefig(output/'rms_Dornheim_db_v2.png',dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print('Outputs:',output.resolve())
    return summary


if __name__ == '__main__':
    main()
