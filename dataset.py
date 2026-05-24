"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

RowGroupRef = Tuple[str, int, int]
RowGroupStats = Tuple[str, int, int, Optional[int], Optional[int]]
TimeWindow = Optional[Tuple[Optional[int], Optional[int]]]


def _epoch_is_millisecond_unit(ts: int) -> bool:
    """Best-effort check for epoch milliseconds rather than seconds."""
    return abs(int(ts)) >= 10 ** 11


def _to_local_datetime(ts: int, tz_offset_hours: int) -> datetime:
    """Convert an epoch timestamp to a timezone-aware local datetime."""
    tz = timezone(timedelta(hours=tz_offset_hours))
    scale = 1000.0 if _epoch_is_millisecond_unit(ts) else 1.0
    return datetime.fromtimestamp(float(int(ts)) / scale, tz=tz)


def _local_day_epoch_window(
    year: int,
    month: int,
    day: int,
    tz_offset_hours: int,
    reference_ts: int,
) -> Tuple[int, int]:
    """Return [start, end) epoch bounds for a local calendar day."""
    tz = timezone(timedelta(hours=tz_offset_hours))
    start_dt = datetime(year, month, day, tzinfo=tz)
    end_dt = start_dt + timedelta(days=1)
    factor = 1000 if _epoch_is_millisecond_unit(reference_ts) else 1
    return int(start_dt.timestamp() * factor), int(end_dt.timestamp() * factor)


def _derive_sample_clock_features(
    timestamps: np.ndarray,
    tz_offset_hours: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build wall-clock feature arrays from sample timestamps.

    Returns:
        day_of_week: np.ndarray of shape (B,), values in [1, 7]  (0 reserved for padding).
        hour_id:     np.ndarray of shape (B,), values in [1, 24] (0 reserved for padding).
        hour_sin:    np.ndarray of shape (B,), continuous cyclic encoding.
        hour_cos:    np.ndarray of shape (B,), continuous cyclic encoding.
    """
    import math
    day_of_week = np.zeros(len(timestamps), dtype=np.int64)
    hour_id = np.zeros(len(timestamps), dtype=np.int64)
    hour_sin = np.zeros(len(timestamps), dtype=np.float32)
    hour_cos = np.zeros(len(timestamps), dtype=np.float32)
    for row_idx, raw_ts in enumerate(timestamps):
        local_dt = _to_local_datetime(int(raw_ts), tz_offset_hours)
        day_of_week[row_idx] = local_dt.weekday() + 1          # Monday=1 ... Sunday=7
        hour_zero_based = local_dt.hour
        hour_id[row_idx] = hour_zero_based + 1                 # 1-24
        hour_angle = 2.0 * math.pi * hour_zero_based / 24.0
        hour_sin[row_idx] = math.sin(hour_angle)
        hour_cos[row_idx] = math.cos(hour_angle)
    return day_of_week, hour_id, hour_sin, hour_cos


def _resolve_valid_day_window(
    valid_date: str,
    reference_ts: int,
    tz_offset_hours: int,
) -> Tuple[int, int, str]:
    """Parse ``MM-DD`` / ``YYYY-MM-DD`` into [start, end) epoch bounds."""
    date_token = valid_date.strip().replace('/', '-').replace('.', '-')
    date_parts = [part for part in date_token.split('-') if part]
    if len(date_parts) == 2:
        ref_dt = _to_local_datetime(reference_ts, tz_offset_hours)
        year = ref_dt.year
        month, day = (int(date_parts[0]), int(date_parts[1]))
    elif len(date_parts) == 3:
        year, month, day = (int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
    else:
        raise ValueError(
            f"Unsupported valid_date={valid_date!r}; expected MM-DD or YYYY-MM-DD"
        )
    start_ts, end_ts = _local_day_epoch_window(
        year=year,
        month=month,
        day=day,
        tz_offset_hours=tz_offset_hours,
        reference_ts=reference_ts,
    )
    return start_ts, end_ts, f"{year:04d}-{month:02d}-{day:02d}"


def _row_group_overlaps_time_window(
    min_ts: Optional[int],
    max_ts: Optional[int],
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> bool:
    """Return whether a row-group timestamp span may overlap a target window."""
    if min_ts is None or max_ts is None:
        return True
    if end_ts is not None and min_ts >= end_ts:
        return False
    if start_ts is not None and max_ts < start_ts:
        return False
    return True


def _parquet_files_from_path(path: str, empty_dir_message: str) -> List[str]:
    """Resolve a directory or file path to the parquet files to scan."""
    if not os.path.isdir(path):
        return [path]

    import glob as _glob
    files = sorted(_glob.glob(os.path.join(path, '*.parquet')))
    if not files:
        raise FileNotFoundError(empty_dir_message)
    return files


def _parquet_files_from_dir(data_dir: str) -> List[str]:
    """Resolve a training data directory to its sorted parquet files."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir}")
    return files


def _row_group_refs_from_files(parquet_files: List[str]) -> List[RowGroupRef]:
    """Return ``(file_path, row_group_index, num_rows)`` triples."""
    row_groups: List[RowGroupRef] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            row_groups.append((
                file_path,
                rg_idx,
                pf.metadata.row_group(rg_idx).num_rows,
            ))
    return row_groups


def _timestamp_column_index(parquet_file: str) -> int:
    """Find the timestamp column index used for row-group statistics."""
    schema_names = pq.ParquetFile(parquet_file).schema_arrow.names
    if 'timestamp' not in schema_names:
        raise KeyError("timestamp column is required for PCVR parquet loading")
    return schema_names.index('timestamp')


def _collect_row_group_stats(parquet_files: List[str]) -> List[RowGroupStats]:
    """Collect row-group refs plus optional min/max timestamp statistics."""
    ts_col_idx = _timestamp_column_index(parquet_files[0])
    row_groups: List[RowGroupStats] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_meta = pf.metadata.row_group(rg_idx)
            min_ts: Optional[int] = None
            max_ts: Optional[int] = None
            try:
                col_meta = rg_meta.column(ts_col_idx)
                stats = col_meta.statistics
                if stats is not None and stats.min is not None and stats.max is not None:
                    min_ts = int(stats.min)
                    max_ts = int(stats.max)
            except Exception:
                min_ts = None
                max_ts = None
            row_groups.append((file_path, rg_idx, rg_meta.num_rows, min_ts, max_ts))
    return row_groups


def _strip_row_group_stats(row_groups: List[RowGroupStats]) -> List[RowGroupRef]:
    """Drop timestamp stats from row-group metadata."""
    return [(file_path, rg_idx, rows) for file_path, rg_idx, rows, _, _ in row_groups]


def _split_by_row_group_tail(
    row_groups: List[RowGroupStats],
    valid_ratio: float,
    train_ratio: float,
) -> Tuple[List[RowGroupRef], List[RowGroupRef], int, int]:
    """Use the historical tail-row-group validation split."""
    total_rgs = len(row_groups)
    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_groups = _strip_row_group_stats(row_groups[:n_train_rgs])
    valid_groups = _strip_row_group_stats(row_groups[n_train_rgs:])
    train_rows = sum(r[2] for r in train_groups)
    valid_rows = sum(r[2] for r in valid_groups)
    logging.info(
        f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
        f"{n_valid_rgs} valid ({valid_rows} rows)"
    )
    return train_groups, valid_groups, train_rows, valid_rows


def _split_by_timestamp_window(
    row_groups: List[RowGroupStats],
    train_ratio: float,
    valid_date: Optional[str],
    valid_start_ts: Optional[int],
    valid_end_ts: Optional[int],
    time_zone_offset_hours: int,
) -> Tuple[List[RowGroupRef], List[RowGroupRef], TimeWindow, TimeWindow, int, int]:
    """Select row groups that can overlap the requested timestamp windows."""
    if valid_start_ts is None and valid_date:
        reference_ts = next(
            (mx for _, _, _, _, mx in reversed(row_groups) if mx is not None),
            None,
        )
        if reference_ts is None:
            raise ValueError(
                "timestamp split requires either row-group timestamp statistics "
                "or explicit --valid_start_ts/--valid_end_ts"
            )
        valid_start_ts, valid_end_ts, canonical_day = _resolve_valid_day_window(
            str(valid_date),
            reference_ts=reference_ts,
            tz_offset_hours=time_zone_offset_hours,
        )
        logging.info(
            "Timestamp split inferred valid day %s in timezone UTC%+d => [%s, %s)",
            canonical_day,
            time_zone_offset_hours,
            valid_start_ts,
            valid_end_ts,
        )

    if valid_start_ts is None or valid_end_ts is None:
        raise ValueError(
            "timestamp split requires --valid_date or both --valid_start_ts and --valid_end_ts"
        )
    if int(valid_end_ts) <= int(valid_start_ts):
        raise ValueError(
            f"Invalid timestamp window: start={valid_start_ts}, end={valid_end_ts}"
        )

    train_time_window: TimeWindow = (None, int(valid_start_ts))
    valid_time_window: TimeWindow = (int(valid_start_ts), int(valid_end_ts))

    train_groups = [
        (file_path, rg_idx, rows)
        for file_path, rg_idx, rows, min_ts, max_ts in row_groups
        if _row_group_overlaps_time_window(
            min_ts, max_ts, train_time_window[0], train_time_window[1]
        )
    ]
    valid_groups = [
        (file_path, rg_idx, rows)
        for file_path, rg_idx, rows, min_ts, max_ts in row_groups
        if _row_group_overlaps_time_window(
            min_ts, max_ts, valid_time_window[0], valid_time_window[1]
        )
    ]

    if train_ratio < 1.0:
        kept = max(1, int(len(train_groups) * train_ratio))
        train_groups = train_groups[:kept]
        logging.info(
            "timestamp split + train_ratio=%s: using first %s train Row Groups",
            train_ratio,
            kept,
        )

    if not train_groups:
        raise ValueError("Timestamp split produced an empty training Row Group set")
    if not valid_groups:
        raise ValueError("Timestamp split produced an empty validation Row Group set")

    train_rows = sum(r[2] for r in train_groups)
    valid_rows = sum(r[2] for r in valid_groups)
    logging.info(
        "Timestamp split: %s train Row Groups before %s, %s valid Row Groups in [%s, %s); "
        "rows shown here are row-group upper bounds before per-row filtering",
        len(train_groups),
        valid_start_ts,
        len(valid_groups),
        valid_start_ts,
        valid_end_ts,
    )
    return train_groups, valid_groups, train_time_window, valid_time_window, train_rows, valid_rows


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        row_groups: Optional[List[Tuple[str, int, int]]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        sample_time_window: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            row_groups: explicit Row Group triples ``(file_path, rg_idx, num_rows)``.
                When provided, this takes precedence over ``row_group_range``.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            sample_time_window: optional sample-level timestamp window
                ``(start_ts_inclusive, end_ts_exclusive)``. Rows outside the
                window are skipped after reading the Row Group.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        self._parquet_files = _parquet_files_from_path(
            parquet_path,
            empty_dir_message=f"No .parquet files in {parquet_path}",
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.sample_time_window = sample_time_window
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        if row_groups is not None:
            self._rg_list = list(row_groups)
        else:
            self._rg_list = _row_group_refs_from_files(self._parquet_files)
            if row_group_range is not None:
                start, end = row_group_range
                self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                batch_dict = self._trim_batch_to_time_window(batch_dict)
                if batch_dict is None:
                    continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _trim_batch_to_time_window(
        self,
        batch_dict: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply the configured sample timestamp window after batch conversion."""
        if self.sample_time_window is None:
            return batch_dict

        start_ts, end_ts = self.sample_time_window
        timestamps = batch_dict['timestamp']
        row_mask = torch.ones_like(timestamps, dtype=torch.bool)
        if start_ts is not None:
            row_mask &= timestamps >= int(start_ts)
        if end_ts is not None:
            row_mask &= timestamps < int(end_ts)

        if bool(row_mask.all()):
            return batch_dict
        if not bool(row_mask.any()):
            return None

        narrowed: Dict[str, Any] = {}
        selected_rows = row_mask.nonzero(as_tuple=False).squeeze(-1)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                narrowed[key] = value.index_select(0, selected_rows)
            elif isinstance(value, list) and len(value) == len(row_mask):
                narrowed[key] = [value[int(i)] for i in selected_rows.tolist()]
            else:
                narrowed[key] = value
        return narrowed

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _fill_int_buffer(
        self,
        batch: "pa.RecordBatch",
        plan: List[Tuple[int, int, int, int]],
        target: np.ndarray,
        stat_group: str,
        B: int,
    ) -> np.ndarray:
        """Fill a pre-allocated integer feature buffer according to a column plan."""
        target[:] = 0
        for ci, dim, offset, vs in plan:
            col = batch.column(ci)
            if dim == 1:
                values = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                values[values <= 0] = 0
                if vs > 0:
                    self._record_oob(stat_group, ci, values, vs)
                else:
                    values[:] = 0
                target[:, offset] = values
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob(stat_group, ci, padded, vs)
                else:
                    padded[:] = 0
                target[:, offset:offset + dim] = padded
        return target

    def _fill_dense_buffer(
        self,
        batch: "pa.RecordBatch",
        plan: List[Tuple[int, int, int]],
        target: np.ndarray,
        B: int,
    ) -> np.ndarray:
        """Fill a pre-allocated dense feature buffer according to a column plan."""
        target[:] = 0
        for ci, dim, offset in plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            target[:, offset:offset + dim] = padded
        return target

    def _fill_sequence_time_buckets(
        self,
        batch: "pa.RecordBatch",
        ts_ci: Optional[int],
        timestamps: np.ndarray,
        target: np.ndarray,
        B: int,
        max_len: int,
    ) -> np.ndarray:
        """Fill pre-allocated sequence time-bucket ids for one domain."""
        target[:] = 0
        if ts_ci is None:
            return target

        ts_col = batch.column(ts_ci)
        ts_offs = ts_col.offsets.to_numpy()
        ts_vals = ts_col.values.to_numpy()
        ts_padded = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            s = int(ts_offs[i])
            e = int(ts_offs[i + 1])
            rl = e - s
            if rl <= 0:
                continue
            ul = min(rl, max_len)
            ts_padded[i, :ul] = ts_vals[s:s + ul]

        ts_expanded = timestamps.reshape(-1, 1)
        time_diff = np.maximum(ts_expanded - ts_padded, 0)
        # Clip raw result to keep the final bucket id inside the Embedding.
        raw_buckets = np.clip(
            np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
            0, len(BUCKET_BOUNDARIES) - 1,
        )
        buckets = raw_buckets.reshape(B, max_len) + 1
        buckets[ts_padded == 0] = 0
        target[:] = buckets
        return target

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._fill_int_buffer(
            batch,
            self._user_int_plan,
            self._buf_user_int[:B],
            'user_int',
            B,
        )

        # ---- item_int ----
        item_int = self._fill_int_buffer(
            batch,
            self._item_int_plan,
            self._buf_item_int[:B],
            'item_int',
            B,
        )

        # ---- user_dense ----
        user_dense = self._fill_dense_buffer(
            batch,
            self._user_dense_plan,
            self._buf_user_dense[:B],
            B,
        )

        # Absolute time features (sample-level).
        sample_day_ids, sample_hour_ids, sample_hour_sin, sample_hour_cos = _derive_sample_clock_features(timestamps, 8)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
            'sample_day_id': torch.from_numpy(sample_day_ids),
            'sample_hour_id': torch.from_numpy(sample_hour_ids),
            'sample_hour_sin': torch.from_numpy(sample_hour_sin),
            'sample_hour_cos': torch.from_numpy(sample_hour_cos),
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._fill_sequence_time_buckets(
                batch,
                ts_ci,
                timestamps,
                self._buf_seq_tb[domain][:B],
                B,
                max_len,
            )

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Supported split modes:
    - ``row_group``: the historical behavior, taking the tail ``valid_ratio``
      fraction of Row Groups as validation.
    - ``timestamp``: train on rows strictly before the validation window and
      validate on rows within ``[valid_start_ts, valid_end_ts)``.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    valid_split_mode = str(kwargs.get('valid_split_mode', 'row_group') or 'row_group')
    valid_date = kwargs.get('valid_date', None)
    valid_start_ts = kwargs.get('valid_start_ts', None)
    valid_end_ts = kwargs.get('valid_end_ts', None)
    time_zone_offset_hours = int(kwargs.get('time_zone_offset_hours', 8))

    pq_files = _parquet_files_from_dir(data_dir)
    row_group_stats = _collect_row_group_stats(pq_files)
    if not row_group_stats:
        raise ValueError(f"No Row Groups found under {data_dir}")

    if valid_split_mode == 'timestamp':
        (
            train_row_groups,
            valid_row_groups,
            train_time_window,
            valid_time_window,
            train_rows,
            valid_rows,
        ) = _split_by_timestamp_window(
            row_group_stats,
            train_ratio=train_ratio,
            valid_date=valid_date,
            valid_start_ts=valid_start_ts,
            valid_end_ts=valid_end_ts,
            time_zone_offset_hours=time_zone_offset_hours,
        )
    else:
        train_time_window = None
        valid_time_window = None
        (
            train_row_groups,
            valid_row_groups,
            train_rows,
            valid_rows,
        ) = _split_by_row_group_tail(
            row_group_stats,
            valid_ratio=valid_ratio,
            train_ratio=train_ratio,
        )

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_groups=train_row_groups,
        clip_vocab=clip_vocab,
        sample_time_window=train_time_window,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_groups=valid_row_groups,
        clip_vocab=clip_vocab,
        sample_time_window=valid_time_window,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(
        f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
        f"batch_size={batch_size}, buffer_batches={buffer_batches}, split_mode={valid_split_mode}"
    )

    return train_loader, valid_loader, train_dataset
