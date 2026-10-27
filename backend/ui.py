"""Built browser shell. This module never reads document storage."""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ASSET_NAME = re.compile(r"[A-Za-z0-9._-]{1,200}")
ASSET_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
}
SHELL_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def ui_root_is_safe(root: Path, storage_root: Path | None) -> bool:
    try:
        resolved = root.resolve(strict=False)
    except OSError:
        return False
    if not resolved.is_dir():
        return False
    if storage_root is None:
        return True
    store = storage_root.resolve(strict=False)
    overlaps = (
        resolved == store or store in resolved.parents or resolved in store.parents
    )
    return not overlaps


def _direct_file(root: Path, relative: Path) -> Path | None:
    if any(part in {"", ".."} for part in relative.parts):
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        if candidate.relative_to(root_resolved) != relative:
            return None
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def asset_file(root: Path, name: str) -> Path:
    if not ASSET_NAME.fullmatch(name) or name in {".", ".."} or ".." in name:
        raise HTTPException(404, "Not found")
    path = _direct_file(root, Path("assets") / name)
    if path is None or path.suffix.lower() not in ASSET_TYPES:
        raise HTTPException(404, "Not found")
    return path


def install_browser(
    app: FastAPI, root: Path | None, *, storage_root: Path | None
) -> bool:
    """Serve the built shell from `/` and `/assets`. Returns False when unavailable."""
    if root is None or not ui_root_is_safe(root, storage_root):
        return False
    index = _direct_file(root, Path("index.html"))
    if index is None:
        return False

    @app.get("/", include_in_schema=False)
    async def browser_shell():
        return FileResponse(
            index,
            media_type="text/html; charset=utf-8",
            content_disposition_type="inline",
            headers=SHELL_HEADERS,
        )

    @app.get("/assets/{name}", include_in_schema=False)
    async def browser_asset(name: str):
        path = asset_file(root, name)
        return FileResponse(
            path,
            media_type=ASSET_TYPES[path.suffix.lower()],
            content_disposition_type="inline",
            headers=SHELL_HEADERS,
        )

    icon = _direct_file(root, Path("favicon.svg"))
    if icon is not None:

        @app.get("/favicon.svg", include_in_schema=False)
        async def browser_icon():
            return FileResponse(
                icon,
                media_type="image/svg+xml",
                content_disposition_type="inline",
                headers=SHELL_HEADERS,
            )

    return True
