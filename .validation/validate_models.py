"""Reproduce switch regression and physics smoke checks; no source mutations."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'deps'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

old = load('draft3_before', HERE / 'draft3_before.py')
new = load('draft3_after', ROOT / 'draft3.py')
core_path = ROOT / 'RTS计算' / 'physics_core.py'
core_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
assert core_hash == 'f0c38ec9a2b3dfc0ecdf9abf04f73b3b6da78beeb94ce618bd43b2f9e4ea0a4b'
core = load('reference_physics_core', core_path)

def functions(path):
    return {n.name: ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}

before = functions(HERE / 'draft3_before.py')
after = functions(ROOT / 'draft3.py')
reference = functions(core_path)
for name in before:
    if name not in ('compute', 'S_ion_chihara'):
        assert before[name] == after[name], name
for name in ('nu_BMA', 'sii_arkhipov'):
    assert after[name] == reference[name], name
assert new.NU_MODE == 'const' and new.SII_MODE == 'dh'

def compare(args, **kwargs):
    a, b = old.compute(*args, **kwargs), new.compute(*args, **kwargs)
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b))
    return max(float(np.max(np.abs(x-y) / np.maximum(np.abs(x), np.finfo(float).tiny))) for x, y in zip(a, b))

regressions = []
for args, kwargs in [((120., -250., 100.), {}), ((5., -45., 45.), {'nw':400, 'method':'integral'}), ((120., -250., 100.), {'method':'maxwell'})]:
    regressions.append({'args':args, 'kwargs':kwargs, 'max_relative_difference':compare(args, **kwargs)})

def configure(m):
    m.Te_eV, m.ne, m.Zf, m.ZA = 10., 2.4e30, 2., 4.
    m.T = m.Te_eV * m.EV / m.kB
    m.wp = np.sqrt(m.ne*m.e**2/(m.eps0*m.me))
    m.kap = np.sqrt(m.ne*m.e**2/(m.eps0*m.kB*m.T))
    m.nu = m.NU_FRAC*m.wp
    m.mu = m.chem_pot(m.T, m.ne)

configure(old)
configure(new)
regressions.append({'args':(160., -400., 400.), 'max_relative_difference':compare((160., -400., 400.))})
# A nonzero externally assigned module-level nu must also be honored exactly.
old.nu = new.nu = 0.03*new.wp
regressions.append({'external_nu_fraction':0.03, 'max_relative_difference':compare((160., -400., 400.))})
old.nu = new.nu = new.NU_FRAC*new.wp

k = new.k_of_theta(160.)
ks = np.logspace(7, 14, 8)
sii = new.sii_arkhipov(ks, new.T, new.ne, new.Zf, new.ZA)
assert np.all(np.isfinite(sii)) and np.all((sii >= 0) & (sii <= 1))
assert abs(sii[-1]-1) < 1e-8
E, w, *_ = new.compute(160., -400., 400.)
w_rep = np.mean(w)
bma = new.nu_BMA(k, w_rep, new.T, new.ne, new.Zf, new.mu)
assert np.isfinite(bma)
reference_bma = core.nu_BMA(k, w_rep, new.T, new.ne, new.Zf, new.mu)
assert reference_bma == bma

smoke = []
for nu_mode in ('const', 'bma'):
    for sii_mode in ('dh', 'arkhipov'):
        new.NU_MODE, new.SII_MODE = nu_mode, sii_mode
        saved_nu = new.nu
        E, w, S, total, k = new.compute(160., -400., 400., method='auto')
        assert new.nu == saved_nu
        assert all(np.all(np.isfinite(a)) for a in (E, w, S, total))
        elastic = new.S_ion_chihara(w, k, new.T, new.mu, new.ne, new.ZA, new.Zf, new.ELASTIC_WIDTH_eV*new.e/new.hbar)
        free = new.Zf*S
        i = np.argmax(free)
        iz = np.argmin(np.abs(E))
        smoke.append({'nu_mode':nu_mode, 'sii_mode':sii_mode, 'n':len(E), 'S_tot_peak':float(np.max(total)), 'elastic_component_peak':float(np.max(elastic)), 'Compton_component_peak':float(free[i]), 'elastic_to_Compton_component_peak_ratio':float(np.max(elastic)/free[i]), 'Compton_energy_eV':float(E[i]), 'S_tot_near_zero_to_at_Compton_ratio':float(total[iz]/total[i]), 'S_tot_min':float(np.min(total)), 'finite':True, 'module_nu_unchanged':True})
        smoke[-1]['Compton_peak_valid'] = bool(nu_mode == 'const' or bma >= 0)
        smoke[-1]['ratio_note'] = 'max(S_ion)/max(Zf*S); BMA negative nu makes this an arithmetic diagnostic, not a physical Compton peak ratio'

# Switching SII_MODE cannot affect the BMA-internal DH factors.
new.SII_MODE = 'dh'
assert new.nu_BMA(k, w_rep, new.T, new.ne, new.Zf, new.mu) == bma
result = {'python':sys.version, 'numpy':np.__version__, 'reference_sha256':core_hash, 'AST_checks':'unchanged existing physics functions; exact reference function ASTs', 'regressions':regressions, 'conditions':{'Te_eV':new.Te_eV, 'T_K':new.T, 'ne_m3':new.ne, 'ZA':new.ZA, 'Zf':new.Zf, 'E0_keV':new.E0_keV, 'theta_deg':160, 'k_m_inv':k, 'k_over_kap':k/new.kap, 'width_eV':new.ELASTIC_WIDTH_eV, 'w_rep_rad_s':w_rep}, 'sii_dh':new.sii_dh(k,new.T,new.ne,new.Zf), 'sii_arkhipov':new.sii_arkhipov(k,new.T,new.ne,new.Zf,new.ZA), 'sii_scan':list(zip(ks.tolist(),sii.tolist())), 'nu_bma_rad_s':bma, 'wp_rad_s':new.wp, 'nu_bma_over_wp':bma/new.wp, 'bma_positive':bool(bma>0), 'const_NU_FRAC':new.NU_FRAC, 'const_nu_rad_s':new.nu, 'smoke':smoke}
result['reference_nu_bma_rad_s'] = reference_bma
result['bma_frequency_scan'] = [{'w_over_wp':fraction, 'nu_over_wp':float(new.nu_BMA(k, fraction*new.wp, new.T, new.ne, new.Zf, new.mu)/new.wp)} for fraction in (0., 0.1, 1., 2.)]
(HERE / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
