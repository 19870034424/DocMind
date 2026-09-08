import uuid
from pathlib import Path

import aiofiles

from config import settings


class FileStore:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.upload_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, doc_id: str, filename: str) -> Path:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        return self.base_dir / f"{doc_id}.{ext}"

    async def save(self, filename: str, content: bytes) -> tuple[str, Path]:
        doc_id = uuid.uuid4().hex[:12]
        file_path = self.get_path(doc_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return doc_id, file_path

    def exists(self, doc_id: str, filename: str) -> bool:
        return self.get_path(doc_id, filename).exists()

    def delete(self, doc_id: str, filename: str) -> bool:
        path = self.get_path(doc_id, filename)
        if path.exists():
            path.unlink()
            return True
        return False

    def read_bytes(self, doc_id: str, filename: str) -> bytes:
        return self.get_path(doc_id, filename).read_bytes()


file_store = FileStore()
