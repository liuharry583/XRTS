"""Independent numerical checks for the measured-SIF implementation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sif_response import MeasuredSIF
import draft
import draft_sif


def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))


class TestMeasuredSIF(unittest.TestCase):
    def test_gaussian_convolution_matches_analytic_solution(self):
        r = np.arange(-24, 24.001, 0.02)
        sif = MeasuredSIF(r, gaussian(r, 0, 3), {})
        x = np.linspace(-80, 80, 8001)
        output = np.linspace(-40, 20, 777)[::-1]
        result, valid = sif.convolve(x, gaussian(x, -12, 4), output)
        np.testing.assert_allclose(result, gaussian(output, -12, 5), atol=2e-7)
        self.assertTrue(valid.all())

    def test_asymmetric_support_has_no_array_center_shift(self):
        # The support midpoint is +6 eV; the true response maximum is +3 eV.
        r = np.arange(-8, 20.001, 0.02)
        sif = MeasuredSIF(r, gaussian(r, 3, 1), {})
        x = np.linspace(-50, 50, 5001)
        impulse = np.zeros_like(x)
        impulse[len(x)//2] = 1/(x[1]-x[0])
        result, _ = sif.convolve(x, impulse, x)
        self.assertAlmostEqual(float(x[result.argmax()]), 3, places=8)
        self.assertAlmostEqual(float(np.trapezoid(result, x)), 1, places=10)
        np.testing.assert_allclose(result, gaussian(x, 3, 1), atol=1e-10)

    def test_load_policies_reflection_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"sif.txt"
            np.savetxt(path, [[-2, -0.1], [-1, 0.5], [0, 1], [1, 0.8], [3, 0]])
            with self.assertRaises(ValueError):
                MeasuredSIF.load(path, negative="error")
            a = MeasuredSIF.load(path, axis="photon-shift")
            b = MeasuredSIF.load(path, axis="energy-loss")
            np.testing.assert_allclose(a.energy, -b.energy[::-1])
            np.testing.assert_allclose(a.response, b.response[::-1])
            self.assertAlmostEqual(float(np.trapezoid(a.response, a.energy)), 1)
            self.assertEqual(a.metadata["negative_points"], 1)
            self.assertEqual(a.metadata["sha256"], b.metadata["sha256"])

    def test_finite_window_mask_and_constant_signal(self):
        x = np.linspace(-30, 30, 601)
        r = np.linspace(-4, 7, 111)
        y = gaussian(r, 1, 1)
        y /= np.trapezoid(y, r)
        sif = MeasuredSIF(r, y, {})
        order = np.arange(len(x))[::-1]
        result, valid = sif.broaden_existing(x[order], np.ones(len(x)))
        np.testing.assert_allclose(result[valid], 1, atol=1e-12)
        self.assertFalse(valid[0])
        self.assertFalse(valid[-1])

    def test_gaussian_engine_agrees_with_original_units_and_components(self):
        cfg = draft.PhysicsConfig(Te_eV=12, ne=3e29, E0_keV=2.96, ZA=4, Zf=2.2,
                                  ELECTRON_METHOD="maxwell", NU_MODE="const", nu_rad_s=5e14,
                                  INSTRUMENT_FWHM_eV=8.7)
        sigma = 8.7/np.sqrt(8*np.log(2))
        r = np.linspace(-8*sigma, 8*sigma, 16001)
        sif = MeasuredSIF(r, gaussian(r, 0, sigma), {})
        original = draft.compute(40, -70, 50, nw=251, config=cfg, normalize=False)
        result = draft_sif.compute_components(40, -70, 50, nw=251, config=cfg, sif=sif)
        np.testing.assert_array_equal(original[0], result["E_eV"])
        np.testing.assert_allclose(original[2], result["S_free_raw"], rtol=1e-10, atol=1e-27)
        scale = original[3].max()
        np.testing.assert_allclose(original[3]/scale, result["S_tot_observed"]/scale,
                                   rtol=3e-4, atol=2e-5)
        np.testing.assert_allclose(result["S_tot_observed"],
            cfg.Zf*result["S_free_observed"]+result["S_elastic_observed"], atol=1e-30)
        self.assertTrue(result["rms_valid_mask"].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
