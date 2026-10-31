"""Read the measured SIF and convolve spectra on the photon-shift energy axis.

The measured kernel is area-normalized, never fitted to a Gaussian.
Negative digitization values are explicitly clipped (default) or rejected.
No baseline subtraction, smoothing, or peak recentering is performed.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent
DEFAULT_SIF = ROOT / "sif_0.txt"


@dataclass
class MeasuredSIF:
    energy: np.ndarray
    response: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path=DEFAULT_SIF, *, axis="energy-loss", negative="clip"):
        """axis='energy-loss' reflects the input axis into draft's E=-hbar*w/e.

        Duplicate digitized coordinates are averaged. Zero outside measured
        support is an explicit finite-window assumption, not measured evidence.
        """
        path = Path(path).resolve()
        if axis not in ("photon-shift", "energy-loss"):
            raise ValueError("SIF axis must be photon-shift or energy-loss")
        if negative not in ("clip", "error"):
            raise ValueError("Negative policy must be clip or error")
        data = np.loadtxt(path, encoding="utf-8-sig", ndmin=2)
        if data.shape[1] != 2 or len(data) < 3 or not np.isfinite(data).all():
            raise ValueError("SIF must contain finite two-column energy[eV]/intensity data")
        x, y = data.T.copy()
        negative_count = int(np.count_nonzero(y < 0))
        if negative_count and negative == "error":
            raise ValueError(f"SIF has {negative_count} negative intensities")
        raw_min = float(y.min())
        if negative == "clip":
            y = np.maximum(y, 0)
        if axis == "energy-loss":
            x = -x
        unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        y = np.bincount(inverse, weights=y) / counts
        x = unique
        if len(x) < 3 or not x[0] < 0 < x[-1]:
            raise ValueError("SIF coordinates must span zero energy shift")
        area = float(np.trapezoid(y, x))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("SIF must have a positive finite area")
        y /= area
        meta = dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest(),
                    input_axis=axis, convolution_axis="photon-shift", energy_unit="eV",
                    negative_policy=negative, negative_points=negative_count,
                    minimum_input_intensity=raw_min, duplicate_points=int(len(data)-len(x)),
                    area_before_normalization=area, area_normalized=True,
                    peak_energy_eV=float(x[y.argmax()]), support_eV=[float(x[0]), float(x[-1])],
                    outside_support="zero", baseline_subtracted=False,
                    smoothing=False, recentered=False)
        return cls(x, y, meta)

    def evaluate(self, energy):
        """Area-normalized continuous response, in 1/eV."""
        return np.interp(energy, self.energy, self.response, left=0, right=0)

    def convolve(self, source_energy, source, output_energy):
        """Integral S(E') R(E-E') dE', using a zero-aligned odd FFT kernel.

        source_energy must increase uniformly. The output retains its supplied
        order. Returns the convolution and a full-measured-support coverage mask.
        """
        x, y, out = map(lambda a: np.asarray(a, float),
                        (source_energy, source, output_energy))
        if (x.ndim != 1 or len(x) < 3 or y.shape != x.shape or out.ndim != 1
                or not all(np.isfinite(a).all() for a in (x, y, out))):
            raise ValueError("Expected finite matching 1D source arrays")
        dx = x[1]-x[0]
        if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-7, atol=1e-10):
            raise ValueError("Convolution source grid must be uniformly increasing")
        if np.any(out < x[0]) or np.any(out > x[-1]):
            raise ValueError("Output coordinates must lie within the source grid")
        radius = int(np.ceil(max(abs(self.energy[0]), abs(self.energy[-1]))/dx)) + 1
        offsets = np.arange(-radius, radius+1)*dx
        kernel = self.evaluate(offsets)
        sampled_area = float(kernel.sum()*dx)
        if not np.isfinite(sampled_area) or sampled_area <= 0:
            raise ValueError("Grid too coarse to resolve SIF")
        kernel /= sampled_area  # numerical quadrature normalization, not peak normalization
        weighted = y.copy()
        weighted[[0, -1]] *= 0.5
        blurred = fftconvolve(weighted, kernel, mode="same")*dx
        valid = ((out-self.energy[-1] >= x[0]) &
                 (out-self.energy[0] <= x[-1]))
        return np.interp(out, x, blurred), valid

    def broaden_existing(self, energy, spectrum, *, step=None):
        """Finite-window legacy spectrum: interpolate holes, zero outside its window.

        No post-convolution peak normalization. Missing exterior physics cannot
        be recovered; the returned mask identifies fully covered convolutions.
        """
        energy, spectrum = np.asarray(energy, float), np.asarray(spectrum, float)
        if (energy.ndim != 1 or len(energy) < 3 or spectrum.shape != energy.shape
                or not np.isfinite(energy).all() or not np.isfinite(spectrum).all()):
            raise ValueError("Expected finite matching 1D spectrum arrays")
        order = np.argsort(energy)
        x, y = energy[order], spectrum[order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("Spectrum energy coordinates must be unique")
        if step is None:
            step = min(np.median(np.diff(x)), np.median(np.diff(self.energy)))
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Step must be finite and positive")
        grid = np.linspace(x[0], x[-1], int(np.ceil((x[-1]-x[0])/step))+1)
        return self.convolve(grid, np.interp(grid, x, y), energy)
