"""
Queue of multiple sensors → DictionaryDetectionReader (per sensor) → FIFOMultiDataFeeder → MultiTargetTracker

Each sensor yields dicts like:
{
  "time": datetime (naive or string/epoch per reader config),
  "x": float, "y": float,                # measurement components (state_vector_fields)
  "sensor_id": "radar_front", "snr": 12  # any metadata fields you want propagated
}

- DictionaryDetectionReader turns those dicts into StoneSoup Detections.
- FIFOMultiDataFeeder merges multiple readers as a single FIFO stream (threaded).
- MultiTargetTracker consumes that stream and outputs tracks.

Added:
- Stable small integer Track IDs via TrackIdMap (T01, T02, …)
- Pretty local timestamp with milliseconds
- Readable per-track line: x, y, vx, vy, speed, length
"""

from __future__ import annotations
import datetime as dt
import math
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Iterable, List, Dict, Any
import numpy as np

# StoneSoup
from stonesoup.reader.generic import DictionaryDetectionReader
from stonesoup.feeder.multi import FIFOMultiDataFeeder
from stonesoup.feeder.time import TimeBufferedFeeder, TimeSyncFeeder  # optional wrappers
from stonesoup.models.transition.linear import (
    ConstantVelocity,
    CombinedLinearGaussianTransitionModel,
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.initiator.simple import SinglePointInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.track import Track


# ----------------------------
# Pretty logging helpers
# ----------------------------


class TrackIdMap:
    """Map StoneSoup's raw track ids (UUIDs) to small stable integers."""

    def __init__(self, start: int = 1):
        self.map: Dict[Any, int] = {}
        self.next_id = start

    def get(self, raw_id: Any) -> int:
        if raw_id not in self.map:
            self.map[raw_id] = self.next_id
            self.next_id += 1
        return self.map[raw_id]


def fmt_ts(ts: dt.datetime) -> str:
    """Local time with milliseconds."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    local = ts.astimezone()  # convert to local machine TZ
    return f"{local:%Y-%m-%d %H:%M:%S}.{int(local.microsecond / 1000):03d}"


def fmt_track_line(track: Track, tid: int) -> str:
    m = np.asarray(track.state.mean).ravel()
    x, vx, y, vy = m[0], m[1], m[2], m[3]
    speed = math.hypot(float(vx), float(vy))
    length = getattr(track, "length", len(track.states))  # age proxy
    return (
        f"[T{tid:02d}] x={x:6.2f}  y={y:6.2f}  "
        f"vx={vx:6.2f}  vy={vy:6.2f}  | speed={speed:5.2f}  len={length:02d}"
    )


# ----------------------------
# 1) Sensor → generator of dicts
# ----------------------------


@dataclass(frozen=True)
class SensorCfg:
    sensor_id: str
    rate_hz: float
    pos_noise_std: float
    phase: float = 0.0
    start: dt.datetime = None  # if None, set at init


class SensorDictStream:
    """Simulates a live sensor producing dict rows for DictionaryDetectionReader.

    Modified: s1 ('radar_front') emits 2 targets per tick; s2 ('radar_side') emits 3 targets per tick.
    Each tick yields multiple dicts with the SAME timestamp so the reader groups them into one set.
    """

    def __init__(self, cfg: SensorCfg):
        object.__setattr__(cfg, "start", cfg.start or dt.datetime.now(dt.timezone.utc))
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        k = 0
        t = self.cfg.start
        dt_s = 1.0 / self.cfg.rate_hz

        # number of targets per sensor
        n_targets = 2 if self.cfg.sensor_id == "radar_front" else 3  # s1=2, s2=3

        # Per-target trajectory params (simple variety per target index)
        # Base offsets differ per sensor to help separation.
        base_x = 10.0 if self.cfg.sensor_id == "radar_front" else 30.0
        base_y = 0.0

        # Precompute per-target parameters
        params = []
        for i in range(n_targets):
            vx = 1.1 + 0.2 * i                 # m/s
            amp = 4.0 + 0.8 * (i % 2)          # sine amplitude
            omega = 0.35 + 0.05 * i            # rad/s
            phase = self.cfg.phase + 0.8 * i   # phase offset
            x0 = base_x + 3.0 * i              # staggered starts
            y0 = base_y + 2.0 * i
            params.append((x0, y0, vx, amp, omega, phase))

        while not self._stop.is_set():
            tsec = k * dt_s

            # Emit ALL targets at the same timestamp t (grouped into one detection set)
            for (x0, y0, vx, amp, omega, phase) in params:
                x = x0 + vx * tsec
                y = y0 + amp * math.sin(omega * tsec + phase)

                # add measurement noise
                meas = np.array([x, y]) + np.random.normal(0.0, self.cfg.pos_noise_std, 2)

                yield {
                    "time": t,  # same timestamp for all targets this tick
                    "x": float(meas[0]),
                    "y": float(meas[1]),
                    "sensor_id": self.cfg.sensor_id,
                    "snr": 20.0 - 5.0 * abs(np.random.normal(0, 1)),
                }

            k += 1
            t = t + dt.timedelta(seconds=dt_s)
            time.sleep(dt_s * 0.9)  # simulate realtime pacing


# ------------------------------------------------------
# 2) Build per-sensor DictionaryDetectionReader instances
# ------------------------------------------------------


def build_dict_reader(dict_iter: Iterable[Dict[str, Any]]) -> DictionaryDetectionReader:
    """
    state_vector_fields: fields used to form the detection vector (in order).
    time_field: key holding the time (datetime, parseable string, or epoch).
    metadata_fields: optional subset to carry into Detection.metadata.
    """
    return DictionaryDetectionReader(
        dictionaries=iter(dict_iter),
        state_vector_fields=("x", "y"),
        time_field="time",
        metadata_fields=("sensor_id", "snr"),
        # If sending strings/epochs instead of datetimes:
        # time_field_format="%Y-%m-%dT%H:%M:%S.%fZ"  or  timestamp=True
    )


# ----------------------------------------
# 3) Merge readers: FIFO (threaded) feeder
# ----------------------------------------


def build_fifo_feeder(
    readers: List[DictionaryDetectionReader],
    *,
    max_queue: int = 512,
    time_ordering: bool = False,
    time_window_s: float | None = None,
) -> FIFOMultiDataFeeder:
    """
    FIFO merges by arrival; if you need strict time ordering, wrap with:
      - TimeBufferedFeeder to buffer & order by time,
      - and optionally TimeSyncFeeder to group detections within a window.
    """
    base = FIFOMultiDataFeeder(readers=readers, max_size=max_queue)

    # TODO: investigate inf loop when using Time sorters.
    # if time_ordering:
    #     base = TimeBufferedFeeder(reader=base, buffer_size=2000)
    # if time_window_s is not None:
    #     base = TimeSyncFeeder(reader=base, time_window=dt.timedelta(seconds=time_window_s))
    return base


# -------------------------------------
# 4) MultiTargetTracker stack (CV + NN)
# -------------------------------------


def build_tracker() -> tuple[MultiTargetTracker, LinearGaussian]:
    q = 1.0  # process noise spectral density; tune 0.1–10+
    # 2D motion = combine two 1D CVs (X and Y), each needs noise_diff_coeff=q
    motion = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(noise_diff_coeff=q),  # X: [x, vx]
            ConstantVelocity(noise_diff_coeff=q),  # Y: [y, vy]
        ]
    )

    predictor = KalmanPredictor(motion)

    # Measure positions [x, y] from the 4D state
    meas_model = LinearGaussian(
        ndim_state=motion.ndim,
        mapping=(0, 2),  # pick x and y
        noise_covar=np.diag([0.5**2, 0.5**2]),  # tune measurement noise
    )
    updater = KalmanUpdater(measurement_model=meas_model)
    hypothesiser = DistanceHypothesiser(
        predictor=predictor, updater=updater, measure=Mahalanobis(), missed_distance=12.0
    )
    associator = NearestNeighbour(hypothesiser)

    # Wide prior; SinglePointInitiator uses updater+measurement_model to fold in first det
    init_state = GaussianState(
        state_vector=StateVector([[0.0], [0.0], [0.0], [0.0]]),
        covar=CovarianceMatrix(np.diag([100.0, 100.0, 100.0, 100.0])),
        timestamp=None,
    )
    initiator = SinglePointInitiator(
        prior_state=init_state, measurement_model=meas_model, updater=updater
    )

    deleter = UpdateTimeStepsDeleter(time_steps_since_update=10)

    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=None,  # set later with feeder
        data_associator=associator,
        updater=updater,
    )
    return tracker, meas_model


# -------------------------
# 5) Wire & run
# -------------------------


def run(runtime_s: float = 12.0):
    # sensors -> dict streams
    start = dt.datetime.now(dt.timezone.utc)
    s1 = SensorDictStream(
        SensorCfg("radar_front", rate_hz=5.0, pos_noise_std=0.7, phase=0.0, start=start)
    )
    s2 = SensorDictStream(
        SensorCfg("radar_side", rate_hz=3.0, pos_noise_std=1.0, phase=1.2, start=start)
    )

    # streams -> DictionaryDetectionReaders
    r1 = build_dict_reader(iter(s1))
    r2 = build_dict_reader(iter(s2))

    # readers -> FIFO feeder (optionally enforce time ordering or sync windows)
    feeder = build_fifo_feeder([r1, r2], max_queue=256, time_ordering=False, time_window_s=None)

    # tracker
    tracker, _ = build_tracker()
    tracker.detector = feeder  # the feeder *is* a DetectionReader

    # pretty id mapper
    idmap = TrackIdMap(start=1)

    t0 = dt.datetime.now(dt.timezone.utc)

    # Iterate over tracker (which is an iterator): each loop internally calls next()
    # on MultiTargetTracker → feeder → readers → sensor streams, pulling one time step
    # of detections through the whole chain and yielding (timestamp, tracks).
    for time_step, tracks in tracker:
        # stop after desired runtime
        if (time_step - t0).total_seconds() > runtime_s:
            s1.stop()
            s2.stop()
            break

        ts_str = fmt_ts(time_step)
        print(f"{ts_str} | tracks={len(tracks):02d} :")
        for tr in sorted(tracks, key=lambda t: idmap.get(t.id)):
            tid = idmap.get(tr.id)
            print(" ", fmt_track_line(tr, tid))


if __name__ == "__main__":
    run(runtime_s=3.0)
