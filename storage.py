import hashlib
import re
from pathlib import Path
from uuid import uuid4


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name)
    return name[:200] or "document"


def save_upload(content: bytes, filename: str, storage_dir: Path) -> tuple[Path, str]:
    digest = hashlib.sha256(content).hexdigest()
    storage_dir.mkdir(parents=True, exist_ok=True)
    target = storage_dir / f"{uuid4().hex}_{safe_filename(filename)}"
    target.write_bytes(content)
    return target, digest
