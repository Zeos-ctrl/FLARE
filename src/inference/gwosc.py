"""
Access to open gravitational-wave data from GWOSC.

Thin wrappers around :mod:`gwosc` and :mod:`gwpy` that fetch strain data and
estimate power spectral densities for a named catalogue event, returning
PyCBC-friendly objects for use by :mod:`src.inference.estimator`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectorData:
    """Conditioned strain and PSD for a single detector."""
    detector: str
    strain: "object"          # pycbc.types.TimeSeries
    psd: "object"             # pycbc.types.FrequencySeries
    delta_t: float
    epoch: float


@dataclass
class EventData:
    """Container for all detector data associated with a GWOSC event."""
    event_name: str
    gps_time: float
    detectors: Dict[str, DetectorData] = field(default_factory=dict)
    f_lower: float = 20.0

    @property
    def detector_names(self) -> List[str]:
        return list(self.detectors.keys())


def get_event_gps(event_name: str) -> float:
    """Return the GPS merger time for a named GWOSC event (e.g. 'GW150914')."""
    from gwosc.datasets import event_gps
    gps = float(event_gps(event_name))
    logger.info("Resolved %s -> GPS %.3f", event_name, gps)
    return gps


def list_events(catalog: Optional[str] = None) -> List[str]:
    """List available GWOSC event names, optionally filtered to one catalog."""
    from gwosc.datasets import find_datasets
    return list(find_datasets(type="events", catalog=catalog))


def fetch_event_data(
    event_name: str,
    detectors: List[str] = ("H1", "L1"),
    duration: float = 32.0,
    sample_rate: float = 4096.0,
    f_lower: float = 20.0,
    psd_seconds: float = 4.0,
    crop_seconds: float = 2.0,
) -> EventData:
    """Fetch and condition open strain data around a GWOSC event.

    Follows the standard pycbc matched-filter recipe (validated against the
    official GW150914 tutorial). For each detector the strain is downloaded,
    FIR high-passed, cropped to remove filter transients, and an on-source PSD
    is estimated and *inverse-spectrum-truncated* so its whitening filter is
    finite-length -- without that truncation the matched filter suffers wrap-
    around leakage that inflates the SNR several-fold and biases mass recovery.

    A long (``duration`` ~32 s) segment is essential: an 8 s segment is too
    short to condition cleanly and produces the same leakage inflation.

    The returned strain is centred on the event GPS time (the merger sits at the
    segment centre after symmetric cropping).

    Args:
        event_name: GWOSC catalogue name, e.g. ``"GW150914"``.
        detectors: Interferometers to fetch (subset of H1/L1/V1).
        duration: Seconds of data centred on the event *before* cropping.
        sample_rate: Target sample rate in Hz (template delta_t = 1/sample_rate).
        f_lower: Low-frequency cutoff for the matched filter (Hz).
        psd_seconds: Welch/PSD segment length in seconds (also the IST length).
        crop_seconds: Seconds trimmed from each end after high-passing.
    """
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.types import TimeSeries as PyCBCTimeSeries

    gps = get_event_gps(event_name)
    delta_t = 1.0 / sample_rate
    half = duration / 2.0
    # FIR high-pass corner sits a little below the analysis f_lower.
    hp_corner = max(10.0, f_lower - 5.0)

    event = EventData(event_name=event_name, gps_time=gps, f_lower=f_lower)

    for det in detectors:
        logger.info("Fetching %s data for %s...", det, event_name)
        ts = GWpyTimeSeries.fetch_open_data(
            det, gps - half, gps + half, sample_rate=int(sample_rate), cache=True,
        )
        strain = PyCBCTimeSeries(ts.value.astype(np.float64), delta_t=delta_t,
                                 epoch=float(ts.t0.value))
        # FIR high-pass + symmetric crop to drop the filter transients. The event
        # GPS stays at the segment centre.
        strain = strain.highpass_fir(hp_corner, 512).crop(crop_seconds, crop_seconds)

        # On-source PSD (the signal is negligible over a ~28 s segment), then
        # interpolate to the strain resolution and inverse-spectrum-truncate.
        seg_len = min(int(psd_seconds * sample_rate), len(strain))
        psd = strain.psd(seg_len * delta_t)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(
            psd, seg_len, low_frequency_cutoff=hp_corner, trunc_method="hann")

        event.detectors[det] = DetectorData(
            detector=det, strain=strain, psd=psd,
            delta_t=delta_t, epoch=float(strain.start_time),
        )
        logger.info("  %s: %d samples @ %.0f Hz (%.1f s)",
                    det, len(strain), sample_rate, len(strain) * delta_t)

    return event
