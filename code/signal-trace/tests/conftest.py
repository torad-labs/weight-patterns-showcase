"""Pytest compatibility shim for python-build-standalone builds lacking os.memfd_create."""
import ctypes
import os
import sys

if sys.platform == "linux" and not hasattr(os, "memfd_create"):
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _SYS_MEMFD_CREATE = 319 if os.uname().machine in ("x86_64", "amd64") else 279

    def _memfd_create(name: str, flags: int = 1) -> int:
        raw = name.encode("utf-8") if isinstance(name, str) else name
        fd = _libc.syscall(_SYS_MEMFD_CREATE, raw, flags)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), name)
        return int(fd)

    os.memfd_create = _memfd_create  # type: ignore[attr-defined]
