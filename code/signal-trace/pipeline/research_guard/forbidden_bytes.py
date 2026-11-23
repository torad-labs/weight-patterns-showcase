"""Stateless UTF-8 prefilter used by the mandatory round-two text scan.

Derived from Performance's by_bytes prototype, not its receipt cache. A False
result means use the original read_text(errors='ignore') and regex search. This
module neither walks files nor changes the verifier's scope/order/size limit.
"""
import locale
import re
import sys


def needles(references):
    """Return necessary ASCII substrings, or no fast path for this method."""
    # The old regex of an empty list matches the empty string, so an empty table
    # must never certify a file clean. An empty reference also requires fallback.
    if not references or (locale.getencoding().lower().replace('-', '').replace('_', '') != 'utf8' and not sys.flags.utf8_mode):
        return None
    table = {}
    for reference in references:
        runs = re.findall(r'[A-Za-z0-9]+', reference) if reference.isascii() and '\r' not in reference and '\n' not in reference else []
        if not runs:
            return None
        table.setdefault(max(runs,key=len).encode('ascii'), []).append(reference.encode('ascii'))
    return table


def without_a_reference(data, table):
    """True only if the original text interpretation cannot contain a reference."""
    if table is None:
        return False
    if not data.isascii():
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            return False
    return not any(reference in data for word,references in table.items() if word in data for reference in references)
