"""Archive compact dose captures without releasing their verified prefix dependencies.

A compact/1 reading is not a smaller whole capture. Its capture holds only the
suffix named by its producer record, while the unchanged prefix stays in the
qualified base. This module keeps those two byte inventories bound through
read-back receipts before it removes a stage's physical suffix.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from pipeline.weight_edit.loop_archive import CaptureArchive


FORM = "compact/1"


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _setup(root, run):
    root = Path(root).resolve()
    raw_run = Path(run)
    if raw_run.is_symlink():
        raise ValueError("the run path is a symlink")
    run = raw_run.resolve()
    studies = root / "studies"
    if not run.is_relative_to(studies):
        raise ValueError("the run is outside this root's studies directory")
    _directory(run, run, "run")
    return root, run


def _inside(path, run, what, *, exists=True):
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        parts = path.relative_to(run).parts
    except ValueError as error:
        raise ValueError(f"{what} is outside the run") from error
    current = run
    for part in parts:
        if part in ("", ".", ".."):
            raise ValueError(f"{what} has an unsafe path")
        current /= part
        if current.is_symlink():
            raise ValueError(f"{what} is a symlink")
    if exists and not current.exists():
        raise ValueError(f"{what} is missing")
    return current


def _record_path(root, run, value, what, *, exists=True):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} has no safe path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{what} has an unsafe path")
    return _inside(root / relative, run, what, exists=exists)


def _directory(path, run, what):
    path = _inside(path, run, what)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"{what} is not a directory")
    return path


def _file(path, run, what):
    path = _inside(path, run, what)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"{what} is not a regular file")
    return path


def _load(path, run, what):
    path = _file(path, run, what)
    try:
        made = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{what} is not readable JSON") from error
    if not isinstance(made, dict):
        raise ValueError(f"{what} is not a JSON object")
    return made


def _write(path, run, value, *, replace=False):
    path = _inside(path, run, "archive receipt", exists=False)
    _directory(path.parent, run, "archive receipt parent")
    if path.is_symlink():
        raise ValueError("archive receipt is a symlink")
    if path.exists() and not replace:
        raise ValueError("archive receipt already exists")
    path.write_text(json.dumps(value, indent=1) + "\n")


def _relative(root, path):
    try:
        return str(Path(path).relative_to(root))
    except ValueError as error:
        raise ValueError("archive path is outside the root") from error


def _snapshot(folder, run, what):
    folder = _directory(folder, run, what)
    files = {}
    for location, directories, names in os.walk(folder, followlinks=False):
        location = Path(location)
        directories.sort()
        names.sort()
        for name in directories:
            child = location / name
            if child.is_symlink():
                raise ValueError(f"{what} contains a symlink")
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError(f"{what} contains an unsafe directory entry")
        for name in names:
            child = location / name
            if child.is_symlink():
                raise ValueError(f"{what} contains a symlink")
            if not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError(f"{what} contains an unsafe file entry")
            files[str(child.relative_to(folder))] = _digest(child)
    return files


def _inventory(value, what):
    if not isinstance(value, dict):
        raise ValueError(f"{what} has no file inventory")
    checked = {}
    for name, sha256 in value.items():
        if not isinstance(name, str) or not isinstance(sha256, str):
            raise ValueError(f"{what} has an unsafe file inventory")
        relative = Path(name)
        if (not name or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)):
            raise ValueError(f"{what} has an unsafe file inventory")
        checked[name] = sha256
    return checked


def _transfer(root, payload, label, action, invoke):
    studies = root / "studies"
    try:
        relative = str(Path(payload).relative_to(studies))
    except ValueError as error:
        raise ValueError("archive payload is outside studies") from error
    invoke("archive-" + action + "-" + label,
           ["torad-research", "traces", action, relative], cpu=True,
           extra=dict(TORAD_RESEARCH_TRACES=str(studies)))


def _pending(root, run, path, payload, files, what):
    if not path.exists():
        return False
    record = _load(path, run, what + " upload receipt")
    if (record.get("path") != _relative(root, payload)
            or record.get("files") != files
            or record.get("uploaded") is not True
            or record.get("remote_bytes_verified") is not False
            or record.get("local_released") is not False):
        raise ValueError(f"{what} changed after upload; local copy retained")
    return True


def _upload(root, run, folder, payload, label, invoke, what):
    files = _snapshot(payload, run, what)
    if not files:
        raise ValueError(f"{what} is empty")
    pending = folder / ("compact-upload.json" if payload.name == "compact" else "capture-upload.json")
    if not _pending(root, run, pending, payload, files, what):
        _transfer(root, payload, label, "upload", invoke)
        _write(pending, run, dict(path=_relative(root, payload), files=files,
                                  uploaded=True, remote_bytes_verified=False,
                                  local_released=False))
    return files


def _archive_receipt(root, run, receipt_path, payload, what, *, retained=False, stack=()):
    receipt_path = _file(receipt_path, run, what + " archive receipt")
    if receipt_path in stack:
        raise ValueError("archive dependencies contain a cycle")
    record = _load(receipt_path, run, what + " archive receipt")
    files = _inventory(record.get("files"), what + " archive receipt")
    if (record.get("path") != _relative(root, payload)
            or record.get("remote_bytes_verified") is not True
            or not isinstance(record.get("local_released"), bool)):
        raise ValueError(f"{what} has no verified archive receipt")

    if payload.is_symlink():
        raise ValueError(f"{what} is a symlink")
    if payload.exists():
        here = _snapshot(payload, run, what)
        CaptureArchive.checked_local(payload, record, here=here)
        if retained and here != files:
            raise ValueError(f"{what} changed after read-back")
    elif record["local_released"] is not True or record.get("retained_files"):
        raise ValueError(f"{what} is missing before local release or lost retained metadata")

    if retained:
        if record["local_released"]:
            raise ValueError(f"{what} was released while a compact stage needs it")
        if not payload.exists():
            raise ValueError(f"{what} is not retained locally")

    dependencies = record.get("dependencies")
    if dependencies is not None:
        _validate_dependencies(root, run, dependencies, stack + (receipt_path,))
    return record


def _identity(root, receipt_path, record):
    return dict(record=_relative(root, receipt_path), sha256=_digest(receipt_path),
                path=record["path"], files=record["files"])


def _validated_identity(root, run, value, what, *, retained=False, stack=()):
    if not isinstance(value, dict):
        raise ValueError(f"{what} has no archive identity")
    receipt_path = _record_path(root, run, value.get("record"), what + " receipt")
    record = _load(receipt_path, run, what + " receipt")
    payload = _record_path(root, run, record.get("path"), what + " payload", exists=False)
    record = _archive_receipt(root, run, receipt_path, payload, what, retained=retained, stack=stack)
    identity = _identity(root, receipt_path, record)
    if value != identity:
        raise ValueError(f"{what} archive identity changed")
    return identity, record


def _base_info(root, run, folder):
    folder = _directory(folder, run, "compact base")
    reading_path = _file(folder / "reading.json", run, "compact base reading")
    reading = _load(reading_path, run, "compact base reading")
    capture = folder / "capture"
    declared_capture = reading.get("capture")
    if (not isinstance(declared_capture, dict)
            or declared_capture.get("folder") != _relative(root, capture)):
        raise ValueError("compact base does not name its scored capture")

    compact = _directory(folder / "compact", run, "filtered base")
    filtered_record = _file(compact / "own.json", run, "filtered base reading")
    qualification_path = _file(compact / "qualification.json", run, "filtered base qualification")
    qualification = _load(qualification_path, run, "filtered base qualification")
    file_sha256 = reading.get("file_sha256")
    if not isinstance(file_sha256, str):
        raise ValueError("compact base has no model identity")
    info = dict(
        folder=_relative(root, folder),
        reading=_relative(root, reading_path),
        reading_sha256=_digest(reading_path),
        file_sha256=file_sha256,
        whole_capture=_relative(root, capture),
        filtered_reading=_relative(root, compact),
        filtered_reading_record_sha256=_digest(filtered_record),
        qualification=_relative(root, qualification_path),
        qualification_sha256=_digest(qualification_path),
        qualified=True,
    )
    if (qualification.get("qualified") is not True
            or qualification.get("whole") != info["whole_capture"]
            or qualification.get("reading") != info["filtered_reading"]
            or qualification.get("reading_record_sha256") != info["filtered_reading_record_sha256"]
            or qualification.get("model_sha256") != info["file_sha256"]):
        raise ValueError("the filtered base is not qualified for this whole capture")
    return info, qualification, capture, compact


def _qualified_against(info, qualification, whole_files):
    if qualification.get("whole_record_sha256") != whole_files.get("own.json"):
        raise ValueError("the filtered base is not qualified against this whole capture")


def _validate_base_descriptor(root, run, descriptor, whole_identity, whole_record, filtered_identity=None):
    if not isinstance(descriptor, dict):
        raise ValueError("archive dependencies omit their compact base")
    folder = _record_path(root, run, descriptor.get("folder"), "compact base")
    info, qualification, _, _ = _base_info(root, run, folder)
    if any(descriptor.get(key) != value for key, value in info.items()):
        raise ValueError("compact base identity changed")
    if whole_identity["path"] != info["whole_capture"]:
        raise ValueError("whole-base archive does not name the qualified capture")
    _qualified_against(info, qualification, whole_record["files"])
    if filtered_identity is not None and filtered_identity["path"] != info["filtered_reading"]:
        raise ValueError("filtered-base archive does not name the qualified reading")
    return info, qualification


def _validate_dependencies(root, run, dependencies, stack):
    if (not isinstance(dependencies, dict)
            or "base" not in dependencies
            or "whole_base" not in dependencies):
        raise ValueError("archive dependencies are unresolved")
    whole = dependencies["whole_base"]
    filtered = dependencies.get("filtered_base")
    whole_identity = whole_record = None
    filtered_identity = None
    if whole is not None:
        whole_identity, whole_record = _validated_identity(
            root, run, whole, "whole-base archive", stack=stack)
    if filtered is not None:
        filtered_identity, _ = _validated_identity(
            root, run, filtered, "filtered-base archive", retained=True, stack=stack)
    descriptor = dependencies.get("base")
    if descriptor is not None:
        if whole_identity is None:
            raise ValueError("archive dependencies omit the verified whole base")
        _validate_base_descriptor(root, run, descriptor, whole_identity, whole_record, filtered_identity)


def _base_full_archive(root, run, folder, capture):
    receipt_path = folder / "capture-archive.json"
    if not receipt_path.exists():
        raise ValueError("the compact stage requires a verified whole-base archive")
    record = _archive_receipt(root, run, receipt_path, capture, "whole-base capture")
    return _identity(root, receipt_path, record), record


def _archive_filtered_base(root, run, folder, label, invoke, verify):
    info, qualification, capture, compact = _base_info(root, run, folder)
    if capture.is_symlink():
        raise ValueError("whole-base capture is a symlink")
    if capture.exists():
        _qualified_against(info, qualification, _snapshot(capture, run, "whole-base capture"))
        whole_identity = whole_record = None
    else:
        whole_identity, whole_record = _base_full_archive(root, run, folder, capture)
        _qualified_against(info, qualification, whole_record["files"])

    receipt_path = folder / "compact-archive.json"
    if receipt_path.exists():
        record = _archive_receipt(root, run, receipt_path, compact, "filtered base", retained=True)
        if whole_identity is None:
            whole_identity, whole_record = _base_full_archive(root, run, folder, capture)
        expected = dict(base=info, whole_base=whole_identity)
        if record.get("dependencies") != expected:
            raise ValueError("filtered-base archive dependencies changed")
        return

    files = _upload(root, run, folder, compact, label + "-compact", invoke, "filtered base")
    if not verify:
        return

    if whole_identity is None:
        whole_identity, whole_record = _base_full_archive(root, run, folder, capture)
    _qualified_against(info, qualification, whole_record["files"])
    _transfer(root, compact, label + "-compact", "verify", invoke)
    if _snapshot(compact, run, "filtered base") != files:
        raise ValueError("filtered base changed during verification")
    again, qualification, _, _ = _base_info(root, run, folder)
    if again != info:
        raise ValueError("filtered-base identity changed during verification")
    whole_identity, whole_record = _base_full_archive(root, run, folder, capture)
    _qualified_against(info, qualification, whole_record["files"])
    _write(receipt_path, run, dict(
        path=_relative(root, compact), files=files, remote_bytes_verified=True,
        local_released=False,
        dependencies=dict(base=info, whole_base=whole_identity),
    ))


def _archive_full(root, run, folder, label, invoke, verify, reading):
    capture = folder / "capture"
    declared = reading.get("capture")
    if not isinstance(declared, dict) or declared.get("folder") != _relative(root, capture):
        raise ValueError("the scored capture is not the folder's capture")
    if capture.is_symlink():
        raise ValueError("the scored capture is a symlink")
    if capture.exists():
        _snapshot(capture, run, "scored capture")

    archive = CaptureArchive(root, run, invoke)
    archive.upload(folder, label)
    if verify:
        archive.verify(folder, label)

    compact = folder / "compact"
    if compact.is_symlink():
        raise ValueError("filtered base is a symlink")
    if compact.exists():
        qualification = compact / "qualification.json"
        if qualification.is_symlink():
            raise ValueError("filtered base qualification is a symlink")
        # A failed/missing optional qualification makes this an ordinary whole
        # reading. Preserve the failed cache; never certify or release it.
        # Once a compact receipt exists, its original binding must still hold.
        if ((folder / "compact-archive.json").exists()
                or qualification.exists() and _load(qualification, run, "filtered base qualification").get("qualified") is True):
            _archive_filtered_base(root, run, folder, label, invoke, verify)


def _single_name(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} is unsafe")
    name = Path(value)
    if name.is_absolute() or len(name.parts) != 1 or value in (".", ".."):
        raise ValueError(f"{what} is unsafe")
    return value


def _stage_physical(root, run, folder, reading):
    capture = folder / "capture"
    details = reading.get("capture")
    if (not isinstance(details, dict)
            or details.get("folder") != _relative(root, capture)
            or details.get("record") != _relative(root, capture / "own.json")):
        raise ValueError("compact stage does not name its physical capture")
    _file(folder / "score.json", run, "compact stage score")
    files = reading.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("capture/own.json"), str):
        raise ValueError("compact stage does not bind capture/own.json")
    expected = {"own.json": files["capture/own.json"]}
    answer_files = details.get("files")
    if not isinstance(answer_files, dict) or not answer_files:
        raise ValueError("compact stage has no physical suffix inventory")
    for answer, listed in answer_files.items():
        answer = _single_name(answer, "compact stage answer identity")
        if not isinstance(listed, dict) or not listed:
            raise ValueError("compact stage has an unsafe suffix inventory")
        for name, sha256 in listed.items():
            name = _single_name(name, "compact stage suffix file")
            if not isinstance(sha256, str):
                raise ValueError("compact stage has an unsafe suffix inventory")
            expected[f"own/{answer}/{name}"] = sha256
    return capture, expected, answer_files


def _check_answer_bindings(info, base, whole_files, answer_files):
    bindings = base.get("whole_capture_of_each_answer")
    if not isinstance(bindings, dict) or set(bindings) != set(answer_files):
        raise ValueError("compact stage does not bind every physical suffix to its whole base")
    for answer in answer_files:
        bound = bindings[answer]
        if not isinstance(bound, dict):
            raise ValueError("compact stage has an unsafe whole-base binding")
        if bound.get("folder") != info["whole_capture"] + "/own/" + answer:
            raise ValueError("compact stage whole-base folder changed")
        for field, file in (
            ("turns_sha256", "turns.npy"),
            ("norms_sha256", "norms.npy"),
            ("start_sha256", "start.npy"),
            ("index_sha256", "index.json"),
        ):
            if bound.get(field) != whole_files.get(f"own/{answer}/{file}"):
                raise ValueError("compact stage whole-base inventory changed")


def _stage_dependencies(root, run, folder, reading, answer_files):
    base = reading.get("base")
    if not isinstance(base, dict):
        raise ValueError("compact stage has no base identity")
    base_folder = _record_path(root, run, base.get("folder"), "compact stage base")
    info, qualification, capture, _ = _base_info(root, run, base_folder)
    if any(base.get(key) != value for key, value in info.items()):
        raise ValueError("compact stage base identity changed")

    whole_identity, whole_record = _base_full_archive(root, run, base_folder, capture)
    source_identity = base.get("whole_capture_archive")
    expected_source = dict(record=whole_identity["record"], sha256=whole_identity["sha256"])
    if source_identity != expected_source:
        raise ValueError("compact stage does not bind the verified whole-base archive")
    _qualified_against(info, qualification, whole_record["files"])
    _check_answer_bindings(info, base, whole_record["files"], answer_files)

    filtered_receipt = base_folder / "compact-archive.json"
    if not filtered_receipt.exists():
        raise ValueError("the compact stage requires a verified filtered-base archive")
    filtered_record = _archive_receipt(
        root, run, filtered_receipt, base_folder / "compact", "filtered base", retained=True)
    filtered_identity = _identity(root, filtered_receipt, filtered_record)
    expected_filtered_dependencies = dict(base=info, whole_base=whole_identity)
    if filtered_record.get("dependencies") != expected_filtered_dependencies:
        raise ValueError("filtered-base archive dependencies changed")

    return dict(base=info, whole_base=whole_identity, filtered_base=filtered_identity)


def _stage_receipt(root, record, reading_path, expected, dependencies):
    if record.get("files") != expected:
        raise ValueError("compact stage physical capture changed")
    if record.get("reading") != dict(path=_relative(root, reading_path), sha256=_digest(reading_path)):
        raise ValueError("compact stage reading identity changed")
    if record.get("dependencies") != dependencies:
        raise ValueError("compact stage dependencies changed")


def _archive_stage(root, run, folder, label, invoke, reading, verify):
    reading_path = folder / "reading.json"
    reading_sha256 = _digest(reading_path)
    capture, expected, answer_files = _stage_physical(root, run, folder, reading)
    dependencies = _stage_dependencies(root, run, folder, reading, answer_files)
    receipt_path = folder / "capture-archive.json"

    if receipt_path.exists():
        record = _archive_receipt(root, run, receipt_path, capture, "compact stage capture")
        _stage_receipt(root, record, reading_path, expected, dependencies)
        if verify and not record["local_released"]:
            if not capture.exists():
                raise ValueError("compact stage is missing before local release")
            CaptureArchive.release(_directory(capture, run, "compact stage capture"), record)
            record = dict(record, local_released=True,
                          retained_files=CaptureArchive.retained(record["files"]))
            _write(receipt_path, run, record, replace=True)
        return

    if capture.is_symlink() or not capture.exists():
        raise ValueError("compact stage has no physical capture to archive")
    files = _snapshot(capture, run, "compact stage capture")
    if files != expected:
        raise ValueError("compact stage physical capture does not match its reading")
    files = _upload(root, run, folder, capture, label, invoke, "compact stage capture")
    if files != expected:
        raise ValueError("compact stage physical capture does not match its reading")
    if not verify:
        return

    _transfer(root, capture, label, "verify", invoke)
    if _snapshot(capture, run, "compact stage capture") != expected:
        raise ValueError("compact stage physical capture changed during verification")
    if _digest(reading_path) != reading_sha256:
        raise ValueError("compact stage reading changed during verification")
    current_dependencies = _stage_dependencies(root, run, folder, reading, answer_files)
    if current_dependencies != dependencies:
        raise ValueError("compact stage dependencies changed during verification")
    record = dict(
        path=_relative(root, capture), files=expected, remote_bytes_verified=True,
        local_released=False, retained_files=CaptureArchive.retained(expected),
        reading=dict(path=_relative(root, reading_path), sha256=_digest(reading_path)),
        dependencies=dependencies,
    )
    _write(receipt_path, run, record)
    CaptureArchive.release(_directory(capture, run, "compact stage capture"), record)
    record["local_released"] = True
    _write(receipt_path, run, record, replace=True)


def archive_dose(root, run, folder, label, invoke, verify=True):
    """Upload a scored dose capture and verify it before any eligible local release.

    Plain whole readings delegate to CaptureArchive. A base with a qualified
    compact payload additionally archives that payload but retains it locally. A
    compact/1 stage archives only its capture directory and releases it only
    after its base's whole and filtered receipts still verify.
    """
    root, run = _setup(root, run)
    folder = _directory(folder, run, "dose folder")
    reading = _load(folder / "reading.json", run, "dose reading")
    form = reading.get("form")
    if form is None:
        _archive_full(root, run, folder, label, invoke, verify, reading)
        return
    if form != FORM:
        raise ValueError("the dose reading has an unknown compact form")
    _archive_stage(root, run, folder, label, invoke, reading, verify)
