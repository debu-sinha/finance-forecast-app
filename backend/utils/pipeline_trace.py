"""
Structured pipeline tracing for debugging forecast quality.

Captures a snapshot at every transformation step in the training pipeline,
storing shape, stats, samples, and step-specific diagnostics. Traces are
stored in memory and retrievable via the /api/debug/pipeline-trace endpoint.
"""

import os
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIPELINE_TRACE_ENABLED = os.getenv("PIPELINE_TRACE_ENABLED", "true").lower() == "true"


@dataclass
class StepSnapshot:
    step_number: int
    step_name: str
    timestamp: str
    shape: list  # [rows, cols]
    columns: list
    date_range: dict  # {"min": str, "max": str}
    target_stats: dict  # {"min", "max", "mean", "std", "null_count", "zero_count"}
    covariate_summary: dict  # {cov_name: {"non_null": N, "non_zero": N}}
    sample_head: list  # First 3 rows as dicts
    sample_tail: list  # Last 3 rows as dicts
    details: dict  # Step-specific info


@dataclass
class PipelineTrace:
    target_col: str = ""
    time_col: str = ""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    enabled: bool = True
    steps: list = field(default_factory=list)

    def __post_init__(self):
        if not PIPELINE_TRACE_ENABLED and self.enabled:
            self.enabled = PIPELINE_TRACE_ENABLED

    def add_step(
        self,
        name: str,
        df: pd.DataFrame | None = None,
        target_col: str | None = None,
        time_col: str | None = None,
        covariates: list | None = None,
        details: dict | None = None,
    ):
        if not self.enabled:
            return

        step_number = len(self.steps)
        tc = target_col or self.target_col
        dc = time_col or self.time_col

        # Handle non-DataFrame steps (metadata only)
        if df is None or len(df) == 0:
            snapshot = StepSnapshot(
                step_number=step_number,
                step_name=name,
                timestamp=datetime.now().isoformat(),
                shape=[0, 0] if df is None else [len(df), len(df.columns)],
                columns=list(df.columns) if df is not None else [],
                date_range={},
                target_stats={},
                covariate_summary={},
                sample_head=[],
                sample_tail=[],
                details=_safe_dict(details),
            )
            self.steps.append(snapshot)
            logger.info(f"[TRACE {self.trace_id}] Step {step_number}: {name} (no data)")
            return

        # Shape
        shape = [len(df), len(df.columns)]
        columns = list(df.columns)

        # Date range
        date_range = {}
        date_col_name = dc if dc in df.columns else ("ds" if "ds" in df.columns else None)
        if date_col_name and date_col_name in df.columns:
            try:
                dates = pd.to_datetime(df[date_col_name], errors="coerce")
                date_range = {
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                    "n_unique": int(dates.nunique()),
                }
            except Exception:
                pass

        # Target stats
        target_stats = {}
        target_name = tc if tc in df.columns else ("y" if "y" in df.columns else None)
        if target_name and target_name in df.columns:
            col = pd.to_numeric(df[target_name], errors="coerce")
            target_stats = {
                "column": target_name,
                "min": _safe_float(col.min()),
                "max": _safe_float(col.max()),
                "mean": _safe_float(col.mean()),
                "std": _safe_float(col.std()),
                "median": _safe_float(col.median()),
                "null_count": int(col.isna().sum()),
                "zero_count": int((col == 0).sum()),
                "total_sum": _safe_float(col.sum()),
            }

        # Covariate summary
        cov_summary = {}
        for cov in (covariates or []):
            if cov in df.columns:
                cov_col = pd.to_numeric(df[cov], errors="coerce")
                cov_summary[cov] = {
                    "non_null": int(cov_col.notna().sum()),
                    "non_zero": int((cov_col != 0).sum()),
                    "unique": int(cov_col.nunique()),
                    "min": _safe_float(cov_col.min()),
                    "max": _safe_float(cov_col.max()),
                }

        # Samples
        sample_head = _df_to_safe_dicts(df.head(3))
        sample_tail = _df_to_safe_dicts(df.tail(3))

        snapshot = StepSnapshot(
            step_number=step_number,
            step_name=name,
            timestamp=datetime.now().isoformat(),
            shape=shape,
            columns=columns,
            date_range=date_range,
            target_stats=target_stats,
            covariate_summary=cov_summary,
            sample_head=sample_head,
            sample_tail=sample_tail,
            details=_safe_dict(details),
        )
        self.steps.append(snapshot)

        # Also log to structured logger for file output
        logger.info(
            f"[TRACE {self.trace_id}] Step {step_number}: {name} | "
            f"shape={shape} | "
            f"dates={date_range.get('min', '?')} to {date_range.get('max', '?')} | "
            f"target: min={target_stats.get('min', '?')}, max={target_stats.get('max', '?')}, "
            f"mean={target_stats.get('mean', '?')}, nulls={target_stats.get('null_count', '?')}"
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "n_steps": len(self.steps),
            "steps": [asdict(s) for s in self.steps],
        }


# --- Module-level trace storage ---

_trace_store: dict[str, dict] = {}
_MAX_STORED_TRACES = 5


def store_trace(trace: PipelineTrace):
    _trace_store[trace.trace_id] = trace.to_dict()
    # Evict oldest if over limit
    while len(_trace_store) > _MAX_STORED_TRACES:
        oldest = next(iter(_trace_store))
        del _trace_store[oldest]


def get_latest_trace(trace_id: str | None = None) -> dict | None:
    if trace_id and trace_id in _trace_store:
        return _trace_store[trace_id]
    if _trace_store:
        return list(_trace_store.values())[-1]
    return None


def list_trace_ids() -> list[str]:
    return list(_trace_store.keys())


# --- Helpers ---

def _safe_float(val) -> float | None:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 4)
    except (TypeError, ValueError):
        return None


def _safe_dict(d: dict | None) -> dict:
    if d is None:
        return {}
    result = {}
    for k, v in d.items():
        try:
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.floating):
                result[k] = _safe_float(v)
            elif isinstance(v, pd.DataFrame):
                result[k] = f"DataFrame({v.shape[0]}x{v.shape[1]})"
            elif isinstance(v, pd.Series):
                result[k] = f"Series(len={len(v)})"
            elif isinstance(v, np.ndarray):
                result[k] = f"ndarray(shape={v.shape})"
            else:
                result[k] = str(v)[:200]
        except Exception:
            result[k] = f"<{type(v).__name__}>"
    return result


def _df_to_safe_dicts(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        safe_row = {}
        for col, val in row.items():
            if pd.isna(val):
                safe_row[str(col)] = None
            elif isinstance(val, (np.integer,)):
                safe_row[str(col)] = int(val)
            elif isinstance(val, (np.floating,)):
                safe_row[str(col)] = _safe_float(val)
            elif isinstance(val, pd.Timestamp):
                safe_row[str(col)] = str(val)
            else:
                safe_row[str(col)] = val
        rows.append(safe_row)
    return rows
