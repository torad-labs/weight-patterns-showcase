"""The recorder's identity up to the name of its library folder.

The recorder (code/signal-trace/recorder/observe-stream) is linked with a DT_RUNPATH: the folder its engine libraries are looked for in, written
into the file as a string. On this machine that string names the checkout; a copy that runs on another machine has the same string rewritten in
place (box-engine.sh in records/development/2026-11-13-rented-solve), so its sha256 is another one although every other byte is the program
that was declared. A run that declares its recorder by sha256 (recorder_revision.passes_read_with) can hold such a copy to the declared program
with this second, exact identity: the sha256 of the file with the bytes of that one string zeroed.

  without_library_folder(path)   the sha256 (hex) of the file with its DT_RUNPATH string's bytes zeroed
  record(path)                   {'sha256', 'sha256_without_library_folder'} of the file, as observe-stream.identity.json keeps it

  python -m pipeline.acquisition.recorder_identity <file>            prints the record of a file
  python -m pipeline.acquisition.recorder_identity <file> <record>   writes it

The string is found through the file's own dynamic section and nothing else is masked: the program header that names the dynamic section, the
one DT_RUNPATH entry, the string table's address and size, the string from the entry's offset to its terminator. Two files with one masked
digest are equal in every byte outside that string (where one string is the shorter, the other file holds zeros after it). Anything that
cannot be read exactly raises: a file that is not a 64-bit little-endian ELF, no dynamic section, no DT_RUNPATH, more than one DT_RUNPATH or a
DT_RPATH beside it, a string table that is not in the file, a string that runs outside its table. The libraries the program loads from that
folder are no part of this identity; a capture names them in its own provenance."""
import hashlib, json, struct, sys
from pathlib import Path

PT_LOAD, PT_DYNAMIC = 1, 2
DT_NULL, DT_STRTAB, DT_STRSZ, DT_RPATH, DT_RUNPATH = 0, 5, 10, 15, 29


class NotRead(ValueError):   # the file's library folder could not be located exactly
    pass                     # (a body of its own, not a docstring alone: the dose payload's sender strips docstrings before a file travels)


def library_folder_span(data):
    """(offset, length) of the DT_RUNPATH string in the file's bytes, its terminator not counted."""
    if len(data) < 64 or data[:4] != b'\x7fELF':
        raise NotRead('not an ELF file')
    if data[4] != 2 or data[5] != 1:
        raise NotRead('not a 64-bit little-endian ELF file')
    phoff, = struct.unpack_from('<Q', data, 32)
    phentsize, phnum = struct.unpack_from('<HH', data, 54)
    if phentsize != 56 or phoff + phentsize*phnum > len(data):
        raise NotRead('the program headers are not in the file as its header says')
    loads, dynamic = [], []
    for i in range(phnum):
        kind, _, offset, vaddr, _, filesz, _, _ = struct.unpack_from('<IIQQQQQQ', data, phoff + i*phentsize)
        if kind == PT_LOAD:
            loads.append((vaddr, offset, filesz))
        elif kind == PT_DYNAMIC:
            dynamic.append((offset, filesz))
    if len(dynamic) != 1:
        raise NotRead(f'{len(dynamic)} dynamic sections')
    start, size = dynamic[0]
    if size % 16 or start + size > len(data):
        raise NotRead('the dynamic section is not in the file as its program header says')
    entries = {}
    for at in range(start, start + size, 16):
        tag, value = struct.unpack_from('<qQ', data, at)
        if tag == DT_NULL:
            break
        entries.setdefault(tag, []).append(value)
    paths = len(entries.get(DT_RUNPATH, [])) + len(entries.get(DT_RPATH, []))
    if not entries.get(DT_RUNPATH):
        raise NotRead('no DT_RUNPATH')
    if paths != 1:
        raise NotRead(f'{paths} library folders named (DT_RUNPATH and DT_RPATH together): one is read, not several')
    if len(entries.get(DT_STRTAB, [])) != 1 or len(entries.get(DT_STRSZ, [])) != 1:
        raise NotRead('the dynamic string table is not named exactly once')
    address, table_size = entries[DT_STRTAB][0], entries[DT_STRSZ][0]
    held = [offset + address - vaddr for vaddr, offset, filesz in loads if vaddr <= address and address + table_size <= vaddr + filesz]
    if len(held) != 1 or held[0] + table_size > len(data):
        raise NotRead('the dynamic string table does not lie in exactly one loaded part of the file')
    table = held[0]
    name = entries[DT_RUNPATH][0]
    if name >= table_size:
        raise NotRead('the library folder\'s name starts outside its string table')
    end = data.find(b'\0', table + name, table + table_size)
    if end < 0:
        raise NotRead('the library folder\'s name runs outside its string table')
    return table + name, end - (table + name)


def without_library_folder(path):
    """The sha256 (hex) of the file with the bytes of its DT_RUNPATH string zeroed, nothing else masked."""
    data = Path(path).read_bytes()
    at, length = library_folder_span(data)
    return hashlib.sha256(data[:at] + b'\0'*length + data[at + length:]).hexdigest()


def record(path):
    data = Path(path).read_bytes()
    at, length = library_folder_span(data)
    return dict(sha256=hashlib.sha256(data).hexdigest(),
                sha256_without_library_folder=hashlib.sha256(data[:at] + b'\0'*length + data[at + length:]).hexdigest(),
                masked='the bytes of the DT_RUNPATH string, located through the dynamic section; nothing else',
                masked_bytes=length, masked_at=at)


if __name__ == '__main__':
    made = record(sys.argv[1])
    if len(sys.argv) > 2:
        json.dump(made, open(sys.argv[2], 'w'), indent=1)
    print(json.dumps(made))
