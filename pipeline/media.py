"""Native code-value/F32 utilities. Heavy dependencies load only when needed."""
from pathlib import Path
import hashlib

def require_f32(values):
    import numpy as np
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Finite native FLOAT32 data required; no automatic conversion')
    return values

def pixel_sha(values):
    import numpy as np
    require_f32(values)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

def bit_exact(actual, expected):
    import numpy as np
    require_f32(actual); require_f32(expected)
    return actual.shape == expected.shape and np.array_equal(
        np.ascontiguousarray(actual).view(np.uint32), np.ascontiguousarray(expected).view(np.uint32))

def read_exr(path):
    import numpy as np
    import OpenImageIO as oiio
    handle = oiio.ImageInput.open(str(path))
    if not handle:
        raise RuntimeError(oiio.geterror())
    try:
        spec = handle.spec()
        formats = list(spec.channelformats) or [spec.format] * spec.nchannels
        if any(str(fmt) != 'float' for fmt in formats):
            raise ValueError('EXR must store native FLOAT32, not HALF')
        values = handle.read_image(format=oiio.FLOAT)
        if values is None:
            raise RuntimeError(handle.geterror())
        require_f32(values)
        return {name: np.ascontiguousarray(values[:, :, i]) for i, name in enumerate(spec.channelnames)}
    finally:
        handle.close()

def write_exr(path, channels, space):
    import numpy as np
    import OpenImageIO as oiio
    path = Path(path)
    if path.exists():
        raise FileExistsError('Existing asset preserved: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(channels)
    values = np.stack([require_f32(channels[n]) for n in names], axis=2)
    spec = oiio.ImageSpec(values.shape[1], values.shape[0], len(names), oiio.FLOAT)
    spec.channelnames = names
    spec.attribute('compression', 'zip')
    spec.attribute('oiio:ColorSpace', space)
    if 'A' in names:
        spec.alpha_channel = names.index('A')
        spec.attribute('oiio:UnassociatedAlpha', 1)
    handle = oiio.ImageOutput.create(str(path))
    if not handle or not handle.open(str(path), spec):
        raise RuntimeError('EXR open failed')
    try:
        if not handle.write_image(values):
            raise RuntimeError(handle.geterror())
    finally:
        handle.close()
    decoded = read_exr(path)
    if list(decoded) != names or not all(bit_exact(decoded[n], channels[n]) for n in names):
        raise ValueError('EXR FLOAT32 channel readback changed values')

def white_alpha(path, alpha):
    import numpy as np
    require_f32(alpha)
    one = np.ones_like(alpha)
    write_exr(path, {'R': one, 'G': one, 'B': one, 'A': alpha}, 'coverage-data')

def map_prior(alpha):
    import cv2
    if require_f32(alpha).shape != (1088, 1920):
        raise ValueError('This profile expects the 1920x1088 prior')
    # Validated4096x2160 source fit has37 top/38 bottom pixels of padding.
    return cv2.resize(alpha[37:1050, :], (4096, 2160), interpolation=cv2.INTER_LINEAR)

def make_trimap(prior):
    import cv2
    import numpy as np
    require_f32(prior)
    def disk(radius):
        y, x = np.indices((2 * radius + 1, 2 * radius + 1))
        return np.uint8((x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2)
    fg = cv2.erode(np.uint8(prior >= .98), disk(48), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    bg = cv2.erode(np.uint8(prior <= .02), disk(9), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    trimap = np.full(prior.shape, 128, np.uint8)
    trimap[bg] = 0; trimap[fg] = 255
    return trimap  # Seed classification only; the final alpha is never eroded/thresholded.

def tile_geometry():
    import numpy as np
    origins_x = list(range(0, 3073, 512)); origins_y = [0, 512, 1024, 1136]
    w = np.sin(np.pi * (np.arange(1024, dtype=np.float64) + .5) / 1024) ** 2
    weight = (w[:, None] * w[None, :]).astype(np.float32)
    norm = np.zeros((2160, 4096), np.float32)
    for y in origins_y:
        for x in origins_x:
            norm[y:y+1024, x:x+1024] += weight
    if not (norm > 0).all():
        raise ValueError('Incomplete normalized overlap coverage')
    return origins_x, origins_y, weight, norm
