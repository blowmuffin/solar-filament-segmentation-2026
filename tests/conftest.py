import numpy as np
from pycocotools import mask as mask_util


def rect(r0, r1, c0, c1):
    """RLE of a 2048x2048 rectangle, rows r0:r1, cols c0:c1 (column-major, as the host expects)."""
    m = np.zeros((2048, 2048), np.uint8)
    m[r0:r1, c0:c1] = 1
    return mask_util.encode(np.asfortranarray(m))
