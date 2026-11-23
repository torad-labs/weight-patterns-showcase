"""Release a reproduced working stream only after its useful coordinates qualify."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from pipeline.native_records.group_observation import publish


def release_stream(raw, kept):
    raw,kept=Path(raw),Path(kept)
    index_path=kept/'index.json';qualification_path=kept/'qualification.json'
    index=json.loads(index_path.read_text());qualification=json.loads(qualification_path.read_text())
    if (not qualification['tokens_equal'] or not qualification['all_score_hashes_equal']
            or index['predictions']!=qualification['predictions']):
        raise ValueError('Working stream must qualify against the same-card natural baseline')
    retained=[kept/'turns.npy',kept/'norms.npy']
    if (not all(p.is_file() for p in retained) or
            sum(p.stat().st_size for p in retained)!=index['bytes']):
        raise ValueError('Complete connected coordinates must remain local before release')
    if any(p.stat().st_mtime_ns>index_path.stat().st_mtime_ns for p in retained):
        raise ValueError('Retained coordinates changed after qualification/export')
    if Path(index['source']).resolve()!=raw.parent.resolve():
        raise ValueError('Working stream does not belong to the qualified export')
    record_path=kept/'stream-compaction.json'
    if not raw.exists():return json.loads(record_path.read_text())
    users=subprocess.run(['fuser','--',str(raw)],capture_output=True,text=True)
    if users.returncode!=1:
        raise ValueError('Working stream still has a user or its use cannot be established')
    stat=raw.stat()
    record=dict(time_ct=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        source=str(raw.resolve()),source_bytes=stat.st_size,released_bytes=0,
        retained_index=str(index_path.resolve()),
        retained_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        qualification=str(qualification_path.resolve()),checkpoint=index['checkpoint'],
        recovery='Retained connected coordinates are the analysis artifact. Other native intermediate outputs require replay of the recorded prompt and checkpoint.',status='prepared')
    publish(record_path,record)
    current=raw.stat()
    if (current.st_ino,current.st_size,current.st_mtime_ns)!=(stat.st_ino,stat.st_size,stat.st_mtime_ns):
        raise ValueError('Working stream changed before release')
    raw.unlink();record.update(status='released',released_bytes=stat.st_size)
    publish(record_path,record)
    return record
