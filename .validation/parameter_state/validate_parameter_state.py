"""Regression checks for the parameter/derived-state refactor."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(ROOT/'RTS_calculating/.mplconfig')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_spectrum_equal(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def main():
    old = load('parameter_before', HERE/'draft3_before.py')
    new = load('parameter_after', ROOT/'draft3.py')
    checks = []
    baseline = np.load(HERE/'baseline.npz')
    for i, (theta, method) in enumerate([(120, 'maxwell'), (5, 'integral')]):
        result = new.compute(theta, -60, 40, nw=36, method=method)
        assert_spectrum_equal(result, [baseline[f'{i}_{j}'] for j in range(5)])
    checks.append('Default Maxwell and integral spectra bitwise unchanged')
    # Reproduce old manual state synchronization, compare every returned value.
    cfg = new.PhysicsConfig(Te_eV=10, ne=6.682357703152599e29, ZA=4, Zf=2,
                            E0_keV=2.96, SII_MODE='arkhipov', NU_FRAC=.03,
                            ELASTIC_WIDTH_eV=2)
    for field in cfg.__dataclass_fields__:
        if field != 'nu_rad_s':
            setattr(old, field, getattr(cfg, field))
    old.T = old.Te_eV*old.EV/old.kB
    old.lam0 = old.h*old.c/(old.E0_keV*1e3*old.EV)
    old.wp = np.sqrt(old.ne*old.e**2/(old.eps0*old.me))
    old.kap = np.sqrt(old.ne*old.e**2/(old.eps0*old.kB*old.T))
    old.nu = old.NU_FRAC*old.wp
    old.mu = old.chem_pot(old.T, old.ne)
    for method in ['integral', 'maxwell']:
        assert_spectrum_equal(old.compute(40,-60,40,nw=36,method=method),
                              new.compute(40,-60,40,nw=36,method=method,config=cfg))
    checks.append('Explicit config equals correctly synchronized legacy engine, both methods')
    before_global = new.current_config()
    new.compute(40,-60,40,nw=36,config=cfg)
    assert new.current_config() == before_global
    checks.append('Explicit per-case configs do not mutate module inputs')
    # Legacy direct assignments must now synchronize automatically.
    for field in cfg.__dataclass_fields__:
        if field != 'nu_rad_s':
            setattr(new, field, getattr(cfg, field))
    new.T = new.lam0 = new.wp = new.kap = new.mu = new.nu = -123
    actual = new.compute(40,-60,40,nw=36,method='maxwell')
    assert_spectrum_equal(actual,new.compute(40,-60,40,nw=36,method='maxwell',config=cfg))
    state = new.build_state(cfg)
    for name in ['T','lam0','wp','kap','mu','nu']:
        assert getattr(new,name) == getattr(state,name)
    expected_k = 4*np.pi*np.sin(np.deg2rad(40)/2)/(new.h*new.c/(2.96e3*new.EV))
    assert actual[-1] == expected_k
    checks.append('Direct base assignments repair stale/corrupted derived globals at compute')
    new.E0_keV = 5
    k5 = new.k_of_theta(40)
    np.testing.assert_allclose(k5/expected_k,5/2.96,rtol=1e-15)
    new.configure(Te_eV=20,ne=1.2e30,E0_keV=2.96,NU_FRAC=.05)
    assert new.lam0 == state.lam0
    assert new.nu == .05*new.wp
    checks.append('k_of_theta and configure refresh geometry and thermodynamics')
    frozen = new.current_config()
    try:
        new.configure(Te_eV=0)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid config accepted')
    assert new.current_config() == frozen
    checks.append('Invalid configuration rejected before mutation')
    fixed = replace(cfg, nu_rad_s=1e14)
    assert new.build_state(fixed).nu == 1e14
    assert new.build_state(replace(fixed,ne=2e30)).nu == 1e14
    checks.append('Explicit fixed collision-frequency override stays fixed')
    # BMA and both ion models remain selectable with isolated inputs.
    for mode in ['dh','arkhipov']:
        bma = replace(cfg,NU_MODE='bma',SII_MODE=mode)
        output = new.compute(40,-60,40,nw=12,method='maxwell',config=bma)
        assert all(np.isfinite(a).all() for a in output)
    checks.append('BMA smoke test finite for both ion modes')
    def functions(path):
        return {n.name:ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n,ast.FunctionDef)}
    a,b = functions(HERE/'draft3_before.py'),functions(ROOT/'draft3.py')
    for name in a.keys()-{'compute','k_of_theta','S_ion_chihara'}:
        assert a[name] == b[name],name
    checks.append('All original physics formula function ASTs preserved')
    # Saved database should be reproducible with its own metadata (not defaults).
    with np.load(ROOT/'database/db_v2/case_r10_T20_th040.npz') as d:
        meta=json.loads(str(d['meta_json']))
        db_config=new.PhysicsConfig(ne=float(d['n_e']),**{n:meta[n] for n in
                  ['Te_eV','ZA','Zf','E0_keV','NU_FRAC','NU_MODE','SII_MODE',
                   'ELASTIC_WIDTH_eV','USE_CHIHARA_ION']})
        result=new.compute(meta['theta_deg'],meta['E_lo'],meta['E_hi'],nw=meta['nw'],
                           method=meta['method'],config=db_config)
        np.testing.assert_array_equal(result[0], d['E_eV'])
        for actual, expected in zip([result[2],result[3],result[4]],
                                    [d['S_free'],d['S_tot'],d['k']]):
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0)
    checks.append('Existing db_v2 representative case reproduced from metadata within 1e-12 relative tolerance')
    (HERE/'results.json').write_text(json.dumps({'checks':checks,'passed':True},indent=2),
                                    encoding='utf-8')
    for check in checks:
        print('PASS:',check)


if __name__ == '__main__':
    main()
