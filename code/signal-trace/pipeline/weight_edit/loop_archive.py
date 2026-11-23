"""Upload and verify captures as separate phases without early local release."""
import hashlib
import json
from pathlib import Path


class CaptureArchive:
    def __init__(self, root, run, invoke):
        self.root, self.run, self.invoke = Path(root), Path(run), invoke

    @staticmethod
    def snapshot(capture):
        files = {}
        for path in sorted(capture.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    files[str(path.relative_to(capture))] = hashlib.file_digest(stream, "sha256").hexdigest()
        return files

    @staticmethod
    def retained(files):
        """Keep discovery records in their original locations, not bulk arrays."""
        return {name: sha for name, sha in files.items()
                if name == "own.json" or (len(Path(name).parts) == 3
                    and Path(name).parts[0] == "own" and Path(name).name == "answer.json")}

    @classmethod
    def checked_local(cls, capture, record, *, here=None):
        """Validate full, partially released or retained-only verified bytes."""
        files = record["files"]
        retained = cls.retained(files)
        if "retained_files" in record and record["retained_files"] != retained:
            raise ValueError("the retained capture inventory changed")
        if here is None:
            here = cls.snapshot(capture)
        if any(files.get(name) != sha for name, sha in here.items()):
            raise ValueError("the capture changed after read-back; local copy retained")
        # Old released receipts predate retention. Their metadata may be absent
        # or restored from the verified inventory; do not rewrite those receipts.
        required = retained if "retained_files" in record or not record["local_released"] else {}
        if any(here.get(name) != sha for name, sha in required.items()):
            raise ValueError("retained capture metadata is missing or changed")
        if record["local_released"] and here != files and any(name not in retained for name in here):
            raise ValueError("the released capture has unexpected local files")
        return here

    @classmethod
    def release(cls, capture, record):
        """Release only verified bulk; interrupted unlinks can resume safely."""
        capture = Path(capture)
        here = cls.checked_local(capture, record)
        kept = cls.retained(record["files"])
        for name in sorted(here):
            if name not in kept:
                (capture / name).unlink()
        if capture.exists():
            directories = sorted((p for p in capture.rglob("*") if p.is_dir()),
                                 key=lambda p: len(p.parts), reverse=True)
            for directory in directories + [capture]:
                if not any(directory.iterdir()):
                    directory.rmdir()

    def capture(self, folder):
        folder = Path(folder)
        capture, archive = folder / "capture", folder / "capture-archive.json"
        verified = False
        if archive.exists():
            record = json.loads(archive.read_text())
            verified = (record.get("remote_bytes_verified") is True
                        and isinstance(record.get("local_released"), bool))
            if not verified:
                raise ValueError("the capture has an invalid verified archive receipt")
            if verified:
                if record.get("path") != str(capture.relative_to(self.root)):
                    raise ValueError("the verified archive names another capture")
                here = self.checked_local(capture, record)
                if record["local_released"] and (here != record["files"]
                        or here == self.retained(record["files"])):
                    return None
        if not capture.exists() and not verified:
            raise ValueError("the capture is neither present nor recorded as verified and released")
        reading = json.loads((folder / "reading.json").read_text())
        expected = self.root / reading["capture"]["folder"]
        if (capture.resolve() != expected.resolve()
                or not capture.resolve().is_relative_to(self.run.resolve())):
            raise ValueError("the capture is outside this run or not the one scored")
        if not (folder / "score.json").exists():
            raise ValueError("the capture has not been scored")
        return capture

    def receipt(self, folder, capture):
        path = Path(folder) / "capture-upload.json"
        if not path.exists():
            raise ValueError("the capture has no completed upload receipt")
        record = json.loads(path.read_text())
        if (record.get("path") != str(capture.relative_to(self.root))
                or record.get("uploaded") is not True
                or record.get("files") != self.snapshot(capture)):
            raise ValueError("the capture changed after upload; local copy retained")
        return record

    def transfer(self, action, capture, label):
        relative = str(capture.relative_to(self.root / "studies"))
        self.invoke("archive-" + action + "-" + label,
                    ["torad-research", "traces", action, relative], cpu=True,
                    extra=dict(TORAD_RESEARCH_TRACES=str(self.root / "studies")))

    def upload(self, folder, label):
        capture = self.capture(folder)
        if capture is None:
            return
        archive = Path(folder) / "capture-archive.json"
        if archive.exists() and json.loads(archive.read_text()).get("remote_bytes_verified") is True:
            # capture() validated the surviving bytes against the verified receipt.
            return
        pending = Path(folder) / "capture-upload.json"
        if pending.exists():
            self.receipt(folder, capture)
            return
        files = self.snapshot(capture)
        if not files:
            raise ValueError("the scored capture is empty")
        self.transfer("upload", capture, label)
        pending.write_text(json.dumps(dict(path=str(capture.relative_to(self.root)),
            files=files, uploaded=True, remote_bytes_verified=False, local_released=False),
            indent=1) + "\n")

    def verify(self, folder, label):
        capture = self.capture(folder)
        if capture is None:
            return
        archive = Path(folder) / "capture-archive.json"
        previous = json.loads(archive.read_text()) if archive.exists() else None
        if previous and previous.get("remote_bytes_verified") is True:
            verified = previous
        else:
            record = self.receipt(folder, capture)
            self.transfer("verify", capture, label)
            if self.snapshot(capture) != record["files"]:
                raise ValueError("the capture changed during verification; local copy retained")
            verified = dict(path=record["path"], files=record["files"],
                            remote_bytes_verified=True, local_released=False,
                            retained_files=self.retained(record["files"]))
            archive.write_text(json.dumps(verified, indent=1) + "\n")
        self.release(capture, verified)
        if not verified["local_released"]:
            verified["local_released"] = True
            verified.setdefault("retained_files", self.retained(verified["files"]))
            archive.write_text(json.dumps(verified, indent=1) + "\n")
