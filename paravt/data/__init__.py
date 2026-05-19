"""Data staging utilities for the ParaVT HuggingFace release.

Two entry points:

- :mod:`paravt.data.sanitize` runs once (release prep): reads the
  CPFS-side parquets, rewrites every absolute video path into a
  relative *sentinel* form keyed to four roots (``longvt_source/*``,
  ``museg/*``, ``selfqa/*``), and writes renamed parquets ready for
  upload to ``ParaVT/ParaVT-Parquet``.

- :mod:`paravt.data.materialize` runs on the user side: given a local
  directory where ``ParaVT/ParaVT-Source`` was downloaded (and its zips
  unpacked), it walks the parquets and prepends that root to every
  sentinel path, producing parquets that ``load_dataset`` (or any
  downstream training/eval pipeline) can read directly.

The sentinel scheme is documented in :func:`paravt.data.sanitize.SENTINEL_RULES`
and round-trips byte-exact through ``sanitize → materialize`` (modulo
filename rename).
"""

__all__ = ["sanitize", "materialize"]
