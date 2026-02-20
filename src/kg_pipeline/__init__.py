from .config import Settings

__all__ = [
    "Settings",
    "build_knowledge_graph",
    "ingest_extraction_dump",
    "answer_with_knowledge_graph",
    "detect_sop_discrepancies",
    "write_discrepancy_dashboard",
]


def build_knowledge_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .pipeline import build_knowledge_graph as _impl

    return _impl(*args, **kwargs)


def ingest_extraction_dump(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .pipeline import ingest_extraction_dump as _impl

    return _impl(*args, **kwargs)


def answer_with_knowledge_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .querying import answer_with_knowledge_graph as _impl

    return _impl(*args, **kwargs)


def detect_sop_discrepancies(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .discrepancy import detect_sop_discrepancies as _impl

    return _impl(*args, **kwargs)


def write_discrepancy_dashboard(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .discrepancy import write_discrepancy_dashboard as _impl

    return _impl(*args, **kwargs)
