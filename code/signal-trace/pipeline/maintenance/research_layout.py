"""Move an artifact on the same filesystem and preserve its historical path."""
import os
from pathlib import Path


def move_and_link(source,destination):
    source,destination=Path(source),Path(destination)
    if source.is_symlink() and source.resolve()==destination.resolve(): return
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    is_directory=source.is_dir();before=source.stat(follow_symlinks=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if before.st_dev!=destination.parent.stat().st_dev:
        raise ValueError('This reorganization only permits same-filesystem renames')
    source.rename(destination)
    try:
        source.symlink_to(os.path.relpath(destination,source.parent),target_is_directory=is_directory)
    except BaseException:
        destination.rename(source)
        raise
    after=destination.stat(follow_symlinks=False)
    if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):
        raise ValueError('Rename did not preserve the original artifact identity')
