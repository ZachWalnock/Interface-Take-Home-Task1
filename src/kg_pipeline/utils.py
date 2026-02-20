from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def normalize_ref(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def make_stable_id(*parts: str, prefix: str = "id") -> str:
    key = "||".join(parts).encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:14]
    return f"{prefix}_{digest}"


def make_uuid(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)
