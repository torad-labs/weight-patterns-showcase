"""Encode already-native resident (output, input) matrices without rerounding."""
import numpy as np


def encoded_resident(values, binding):
    """Return the exact GGUF bytes, accepting finite native BF16 values only.

    C order of the resident (output, input) matrix is F order of the existing
    encoder's transposed (input, output) input. Strided and endian-marked FP32
    arrays are read in their logical order; the output is always little-endian.
    Nonzero BF16 subnormals are rejected because the existing engine encoder
    flushes them to signed zero and its exact-value validation rejects them.
    """
    if binding.get('native_type') != 'bf16':
        raise ValueError('the declared native tensor must be BF16')
    shape = binding.get('shape')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 4
            or any(not isinstance(n, (int, np.integer)) or isinstance(n, bool)
                   or n <= 0 for n in shape) or tuple(shape[2:]) != (1, 1)):
        raise ValueError('the declaration must describe one complete BF16 matrix')
    if (not isinstance(values, np.ndarray) or values.dtype.kind != 'f'
            or values.dtype.itemsize != 4 or values.ndim != 2
            or values.shape != (shape[1], shape[0])):
        raise ValueError('resident values must be the declared (output, input) FP32 matrix')
    size = binding.get('bytes')
    if (not isinstance(size, (int, np.integer)) or isinstance(size, bool)
            or size != values.size * 2):
        raise ValueError('the declared BF16 byte count differs from the complete matrix')

    # Same-itemsize view: no float copy, including for transposed/strided input.
    bits = values.view(np.dtype('u4').newbyteorder(values.dtype.byteorder))
    stored = np.empty(values.shape, dtype='<u2', order='C')
    # Reuse the necessary uint16 output for low-half validation, then fill it
    # with the high halves. Neither operation needs a full uint32/uint64 copy.
    np.copyto(stored, bits, casting='unsafe')
    if np.any(stored):
        raise ValueError('FP32 values contain nonzero bits below native BF16 precision')
    np.right_shift(bits, 16, out=stored, casting='unsafe')
    magnitude = stored & np.uint16(0x7fff)
    if np.any(magnitude >= 0x7f80):
        raise ValueError('native BF16 values must be finite')
    if np.any((magnitude > 0) & (magnitude < 0x80)):
        raise ValueError('nonzero BF16 subnormals do not survive the existing encoder')
    del magnitude
    return stored.tobytes(order='C')
