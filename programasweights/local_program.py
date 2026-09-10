"""Import current GGUF .paw bundles without modifying their source or Hub cache.

Only inert program assets are imported. Runtime-generated prefix state is never
accepted from an archive, but an existing local cache's own state is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import cache, client, config


_REQUIRED_FILES = frozenset(("meta.json", "adapter.gguf", "prompt_template.txt"))
_ALLOWED_FILES = _REQUIRED_FILES | {"pseudo_program.txt"}
_COPY_CHUNK_BYTES = 64 * 1024


def _fingerprint(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_fingerprint(info: os.stat_result) -> tuple:
    if os.name != "nt":
        return _fingerprint(info)
    # CPython on Windows can expose creation time as stat(path).st_ctime,
    # but metadata-change time as fstat(fd).st_ctime. Compare the fields with
    # matching meanings across APIs; each API's full fingerprint (including
    # ctime) is still checked against its own baseline after reading.
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
        getattr(info, "st_birthtime_ns", None),
    )


@contextmanager
def _stable_regular_file(path: Path, *, allow_symlink: bool = False):
    """Open without FIFO blocking and reject replacements or concurrent writes."""
    before = path.stat() if allow_symlink else path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Local program asset is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    if not allow_symlink:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or _open_fingerprint(opened) != _open_fingerprint(before):
            raise ValueError(f"Local program asset changed while opening: {path}")
        yield source, before
        after = os.fstat(source.fileno())
        current = path.stat() if allow_symlink else path.lstat()
        if _fingerprint(opened) != _fingerprint(after) or _fingerprint(before) != _fingerprint(current):
            raise ValueError(f"Local program asset changed while reading: {path}")


def _copy_snapshot(path: Path, snapshot: Path) -> str:
    """Hash precisely the bounded bytes copied for subsequent extraction."""
    limit = client.MAX_PAW_ARCHIVE_BYTES
    digest = hashlib.sha256()
    copied = 0
    # Source symlinks are useful (e.g. latest.paw) and safe to read: the opened
    # regular target and the path's target must remain the same during copying.
    with _stable_regular_file(path, allow_symlink=True) as (source, before):
        if before.st_size > limit:
            raise ValueError(f"Local .paw archive exceeds {limit} bytes: {path}")
        with snapshot.open("xb") as output:
            while True:
                chunk = source.read(min(_COPY_CHUNK_BYTES, limit - copied + 1))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > limit:
                    raise ValueError(f"Local .paw archive exceeds {limit} bytes: {path}")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if copied != before.st_size:
            raise ValueError(f"Local .paw archive changed while copying: {path}")
    return digest.hexdigest()


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"Unsafe local program cache directory: {path}")


def _file_digest(path: Path, *, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with _stable_regular_file(path) as (source, before):
        if before.st_size > limit:
            raise ValueError(f"Local program cache asset exceeds expected size: {path}")
        while True:
            chunk = source.read(min(_COPY_CHUNK_BYTES, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"Local program cache asset exceeds expected size: {path}")
            digest.update(chunk)
        if size != before.st_size:
            raise ValueError(f"Local program cache asset changed while reading: {path}")
    return size, digest.hexdigest()


def _validate_extracted(directory: Path) -> tuple[str, dict[str, tuple[int, str]]]:
    names = {entry.name for entry in directory.iterdir()}
    if not _REQUIRED_FILES <= names or not names <= _ALLOWED_FILES:
        raise ValueError(
            "Unsupported local .paw bundle members; require meta.json, adapter.gguf, "
            "and prompt_template.txt, with optional pseudo_program.txt only."
        )
    for name in names:
        if not stat.S_ISREG((directory / name).lstat().st_mode):
            raise ValueError(f"Local .paw bundle member must be a regular file: {name}")
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Local .paw bundle contains invalid meta.json") from error
    program_id = meta.get("program_id") if isinstance(meta, dict) else None
    if not isinstance(program_id, str) or not cache.is_program_id(program_id):
        raise ValueError("Local .paw bundle must contain a valid meta.program_id")
    if not cache.validate_program_assets_dir(directory, program_id):
        raise ValueError("Local .paw bundle contains invalid GGUF program assets")
    expected = {
        name: _file_digest(directory / name, limit=client.MAX_PAW_EXPANDED_BYTES)
        for name in sorted(names)
    }
    return program_id, expected


def import_local_program(path: Path) -> Path:
    """Validate and atomically cache a current-format local GGUF .paw archive.

    The SHA-256 of a stable source snapshot selects ``local_programs/<digest>``.
    A claimed program ID never selects a Hub cache location. Existing immutable
    files must match this same snapshot; conflicting cache data is not replaced.
    No network, runtime-manifest hydration, or base-model cache access occurs.
    """
    path = Path(path)
    # Preserve useful missing-file/permission exceptions before creating cache
    # state. The snapshot rechecks the opened target and bounds every read.
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Local .paw source is not a regular file: {path}")
    if info.st_size > client.MAX_PAW_ARCHIVE_BYTES:
        raise ValueError(f"Local .paw archive exceeds {client.MAX_PAW_ARCHIVE_BYTES} bytes: {path}")

    root = config.get_cache_dir() / "local_programs"
    _ensure_directory(root)
    locks = root / ".locks"
    _ensure_directory(locks)
    # This context owns only its fresh staging directory, never an existing
    # program cache, source path, or shared runtime/model directory.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=str(root)) as temporary:
        staging = Path(temporary)
        snapshot = staging / "source.paw"
        archive_hash = _copy_snapshot(path, snapshot)
        with snapshot.open("rb") as source:
            if source.read(4).startswith(b"PAW"):
                raise ValueError(
                    "Unsupported legacy binary .paw format; local loading requires "
                    "a current GGUF ZIP bundle."
                )
        extracted = staging / "program"
        extracted.mkdir()
        client.PAWClient._safe_extract_paw(snapshot, extracted)
        program_id, expected = _validate_extracted(extracted)

        destination = root / archive_hash
        lock_path = locks / f"{archive_hash}.lock"
        if os.path.lexists(lock_path) and not stat.S_ISREG(lock_path.lstat().st_mode):
            raise ValueError(f"Unsafe local program cache lock: {lock_path}")
        with cache._cross_process_lock(lock_path):
            if os.path.lexists(destination):
                if not cache.validate_program_assets_dir(destination, program_id):
                    raise ValueError(f"Existing local program cache is invalid; retaining it: {destination}")
                for name, expected_digest in expected.items():
                    if _file_digest(destination / name, limit=expected_digest[0]) != expected_digest:
                        raise ValueError(
                            f"Existing local program cache differs from source; retaining it: {destination / name}"
                        )
                return destination
            # Cooperative importers hold the same content-addressed lock. Never
            # move aside or replace an already-existing cache directory.
            os.rename(extracted, destination)
            return destination
