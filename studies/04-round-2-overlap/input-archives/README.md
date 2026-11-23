# Input archives

One `<name>.json` here stands for a declared input of a round-two run whose bytes are in the bucket, by Astra's contract
(`records/retrospectives/2026-11-13-declared-input-archives-astra.md`, and its four boundary repairs of November 13). The record's form
and what the verifier and a native run check are at the top of `code/signal-trace/pipeline/research_guard/input_archives.py`; the
commands (`make`, `files`, `upload`, `verify`, `release`, `restore`) are `archive_inputs.py` beside it.

- A record is an exact inventory: the input's path, the bucket location, and every file's path, bytes and sha256. It stands for a
  missing file only once the bucket's bytes of every inventoried file have been read back and bound to that inventory (`verify`).
- A record states its membership: which answers its files hold, and the unsealed index or declaration that says so. A file whose
  answers the membership does not resolve, or that pools an answer no fit uses, is not inventoried and stays on this disk. The
  membership's sources stay here too.
- `excluded.json` names what no record may hold and what `make` never lists or opens: anything under a sealed name (`reserved`), and
  anything whose path names an answer no fit uses.
- `make` settles whether a path may be an input from its spelling and the way it resolves, before it lists or reads anything.
- The bucket is asked for the inventory's files one by one and never for a folder, when sending, reading back and fetching alike.
- The verifier accepts a declared input whose files are here with their bytes or gone with a verified record. A native run starts only
  when its own declared inputs are back on this disk and held to their inventory (`archive_inputs.py restore <name> --fetch`); the
  receipt of that (`.restored/`, local) holds each file's device, inode, size and both times, and ends when any of them changes.
- `verifications/<name>.txt` is what the bucket tool said when a record was bound.
- `release` is the one command that removes anything: files of a bound inventory, each by its own path, each still the file the
  inventory hashed (its receipt's device, inode, size and both times). It removes nothing when the record does not load whole, when no
  read-back is bound or the registry does not keep its text, or when any inventoried file here is another file. What it took is in
  `releases/<name>.json`. It is run only for what Astra has named (`STATE.md`).

The first three records (November 13, 2026) are round one's observation data, which only the finished run `overlap` declares: Astra
approved the files its three membership statements resolve (`records/retrospectives/2026-11-13-declared-input-archives-astra.md`).
Which inputs only finished runs declare is in the speed review's `declared-inputs/declared-inputs.txt`; which answers each file of
that data holds is in `declared-inputs/observation-membership.txt` beside it.
