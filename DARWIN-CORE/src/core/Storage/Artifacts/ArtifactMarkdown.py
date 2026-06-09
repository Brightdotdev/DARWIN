from __future__ import annotations

from typing import Any

from src.core.Storage.StoreUtils import normalize_record
from src.core.Storage.Transanctions import write_text_atomic
from src.core.Logger.DarwinLogger import darwin, LogType


class ArtifactMarkdown:
    """Markdown artifact I/O and deletion."""

    def has_markdown(self, filename: str) -> bool:
        return self._artifacts.has_artifact(self._resolve_hash(filename), "markdown")

    def save_markdown(self, filename: str, doc: Any) -> None:
        if isinstance(doc, dict):
            error = doc.get("error")
            display_text = doc.get("display_text", "")
            embedding_text = doc.get("embedding_text", "")
        else:
            error = getattr(doc, "error", None)
            display_text = getattr(doc, "display_text", "")
            embedding_text = getattr(doc, "embedding_text", "")

        if error:
            darwin.log(f"Skipping markdown save for {filename}: {error}", LogType.INFO)
            return

        h = self._resolve_hash(filename)
        self._artifacts.ensure_dir(h)

        md_path = self.paths.markdown_path(h)
        display_md_path = self.paths.display_markdown_path(h)

        md_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            write_text_atomic(display_md_path, display_text)
            write_text_atomic(md_path, embedding_text)

            doc_dict = normalize_record(doc)
            skip = {"text", "error", "filename", "images", "embedding_text", "display_text"}
            extras = {k: v for k, v in doc_dict.items() if k not in skip}

            self._meta.update(h, "markdown", extras=extras)
            self._contents.set_stage(h, "markdown")
            self._cache.invalidate(f"markdown:{h}")
            darwin.log(f"Saved markdown for {filename}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Failed to save markdown for {filename}: {e}", LogType.ERROR)

    def load_markdown(self, filename: str) -> str | None:
        h = self._resolve_hash(filename)
        path = self.paths.markdown_path(h)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            darwin.log(f"Error reading markdown for {filename}: {e}", LogType.ERROR)
            return None

    def delete_markdown(self, filename: str) -> bool:
        """Remove markdown artifacts and clear the registry flag."""
        h = self._resolve_hash(filename)
        removed = False
        for p in (self.paths.markdown_path(h), self.paths.display_markdown_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, "markdown", complete=False)
        self._cache.invalidate(f"markdown:{h}")
        return removed