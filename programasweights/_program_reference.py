"""Deterministic local-file intent, independent of filesystem contents."""

from __future__ import annotations

import os
import re
from pathlib import Path


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def local_program_path(reference: object) -> Path | None:
    """Return an explicit local path, or None for an ID/slug/Program.

    Path-like objects always mean files. Strings select files by syntax, never
    by existence; in particular, an ``owner/slug`` remains a Hub reference.
    """
    is_pathlike = isinstance(reference, os.PathLike)
    if is_pathlike:
        value = os.fspath(reference)
        if not isinstance(value, str):
            raise TypeError("Local program paths must be text, not bytes.")
    elif isinstance(reference, str):
        value = reference
    else:
        return None

    if "\x00" in value:
        raise ValueError("Program references cannot contain null bytes.")
    windows_drive = bool(_WINDOWS_DRIVE.match(value))
    if not is_pathlike and not windows_drive and _URI_SCHEME.match(value):
        raise ValueError(
            "Program URLs are not supported. Download the .paw bundle first "
            "and pass its local path."
        )

    explicit_path = value.startswith(
        ("./", "../", "/", "~/", "\\", ".\\", "..\\", "~\\")
    )
    if not (
        is_pathlike or windows_drive or explicit_path
        or value.lower().endswith(".paw")
    ):
        return None
    if not value:
        raise ValueError("Local program path cannot be empty.")

    # A Windows-looking string must never be sent to the Hub on another OS.
    # Path objects, however, explicitly name a native path and may contain
    # characters that would have a different meaning on another platform.
    windows_path = windows_drive or value.startswith(
        ("\\", ".\\", "..\\", "~\\")
    )
    if not is_pathlike and os.name != "nt" and windows_path:
        raise ValueError(
            "This is a Windows filesystem path, which cannot be opened on "
            "this platform. Pass a native local path instead."
        )
    return Path(value).expanduser()
