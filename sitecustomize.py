"""
sly_tqdm patch
"""
try:
    from supervisely.task import progress as _sly_progress  # type: ignore

    _cls = getattr(_sly_progress, "tqdm_sly", None)
    if _cls is not None and not hasattr(_cls, "mininterval"):
        _cls.mininterval = 0.1
except Exception:
    pass

