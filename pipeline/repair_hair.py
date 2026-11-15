"""Remove residual blue sky from a tracked dark-hair gap in straight RGBA.

Rule schema::

    {"type": "hair-sky", "frames": [0, 64],
     "analysis_size": [1080, 570],
     "anchors": [[0, 439, 105], [64, 655, 99]],
     "params": {"search_radius": [91, 84], "ellipse_radius": [89, 80],
                "sky_sample_half_height": 65, "sky_sample_half_width": 235}}

Frames and anchor times are zero-based local indices; the frame range is
inclusive. Anchor x/y coordinates and all lengths are in the unpadded analysis
grid. Centers interpolate linearly between anchors. ``params`` is optional;
the example shows its defaults. The fixed color/distance thresholds below are
calibrated to RGB8 BT.709 code values and this analysis scale, not linear light.

This is a local color-mixture heuristic, not a general hair matte or tracker.
It requires dark opaque hair near a blue/green sky island, an unoccluded sky
sample, and a manually supplied track. Missing evidence produces no change.
The analysis planes are derived from the supplied native RGB/F/A; no editor,
render, model, temporal interpolation of matte pixels, or file access is used.
Only selected native pixels are assigned, preserving all other channel bits.
"""
import math

from .media import require_f32


DEFAULT_PARAMS = {
    'search_radius': [91, 84],
    'ellipse_radius': [89, 80],
    'sky_sample_half_height': 65,
    'sky_sample_half_width': 235,
}


def _smooth(value):
    import numpy as np
    t = np.clip(value, 0, 1)
    return t*t*(3-2*t)


def _to8(values):
    import numpy as np
    return np.floor(np.clip(values, 0, 1)*255+.5).astype(np.uint8)


def _geometry(rule, local, width, height):
    """Validate portable settings and align the search ROI to native pixels."""
    import numpy as np
    if rule.get('type') != 'hair-sky':
        raise ValueError('Expected a hair-sky rule')
    size = rule.get('analysis_size', [1080, 570])
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or
            any(type(n) is not int or n <= 0 for n in size)):
        raise ValueError('analysis_size must be two positive integers')
    aw, ah = size
    if aw > width or ah > height:
        raise ValueError('The hair analysis grid must not upsample the native grid')
    anchors = np.asarray(rule.get('anchors', []), dtype=np.float64)
    if (anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 1 or
            not np.isfinite(anchors).all()):
        raise ValueError('anchors must contain finite [local_frame, x, y] rows')
    if (np.any(anchors[:, 0] < 0) or
            np.any(anchors[:, 0] != np.floor(anchors[:, 0])) or
            np.any(np.diff(anchors[:, 0]) <= 0)):
        raise ValueError('Anchor frames must be unique increasing nonnegative integers')
    if (np.any(anchors[:, 1:] < 0) or np.any(anchors[:, 1] >= aw) or
            np.any(anchors[:, 2] >= ah)):
        raise ValueError('Anchor coordinates must be inside the analysis grid')
    if type(local) is not int or local < 0:
        raise ValueError('Expected a nonnegative local frame index')
    params = dict(DEFAULT_PARAMS)
    supplied = rule.get('params', {})
    if not isinstance(supplied, dict) or set(supplied)-set(params):
        raise ValueError('Unknown hair-sky parameters')
    params.update(supplied)
    for key in ('search_radius', 'ellipse_radius'):
        value = np.asarray(params[key], dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(key+' must contain two finite positive lengths')
        params[key] = value
    if np.any(params['search_radius'] < params['ellipse_radius']):
        raise ValueError('search_radius must enclose ellipse_radius')
    for key in ('sky_sample_half_height', 'sky_sample_half_width'):
        if type(params[key]) is not int or params[key] <= 0:
            raise ValueError(key+' must be a positive integer')
    cx = float(np.interp(local, anchors[:, 0], anchors[:, 1]))
    cy = float(np.interp(local, anchors[:, 0], anchors[:, 2]))
    rx, ry = params['search_radius']
    # For 4096x2160 -> 1080x570 these steps are 135 and 19. Aligning
    # boundaries makes the analysis/native resampling footprint unambiguous.
    sx, sy = aw//math.gcd(width, aw), ah//math.gcd(height, ah)
    x0 = max(0, math.floor((cx-rx)/sx)*sx)
    x1 = min(aw, math.ceil((cx+rx)/sx)*sx)
    y0 = max(0, math.floor((cy-ry)/sy)*sy)
    y1 = min(ah, math.ceil((cy+ry)/sy)*sy)
    return (aw, ah), (cx, cy), (x0, y0, x1, y1), params


def apply_hair(channels, rgb, local, rule):
    """Return corrected straight FLOAT32 R/G/B/A planes, without mutating inputs.

    ``rgb`` is the original full-grid uint8 HxWx3 image. ``channels`` contains
    finite native FLOAT32 planes in restored BT.709 code values with coverage
    A in [0, 1]. The dispatcher selects the rule's inclusive ``frames`` range.
    If no supported sky/hair mixture is found, return ``channels`` itself.
    """
    import cv2
    import numpy as np
    if set(channels) != set('RGBA'):
        raise ValueError('hair-sky requires straight R/G/B/A channels')
    alpha = require_f32(channels['A'])
    if alpha.ndim != 2 or not alpha.size or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError('Invalid native coverage')
    for values in channels.values():
        if require_f32(values).shape != alpha.shape:
            raise ValueError('Hair repair channels have different grids')
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.shape != (*alpha.shape, 3)):
        raise ValueError('Hair repair requires the original full-grid RGB8 image')
    height, width = alpha.shape
    size, (cx, cy), (x0, y0, x1, y1), params = _geometry(rule, local, width, height)
    aw, ah = size
    foreground = np.stack([channels[c] for c in 'RGB'], axis=2)
    # Recreate the unpadded analysis planes directly, including the original
    # round-half-up RGB8 quantization. Premultiplication is for analysis only.
    original = _to8(cv2.resize(rgb.astype(np.float32)/np.float32(255), size,
                               interpolation=cv2.INTER_AREA)).astype(np.float32)
    analysis_a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    matte = _to8(analysis_a)
    associated = cv2.resize(foreground*alpha[:, :, None], size,
                            interpolation=cv2.INTER_AREA)
    composite = _to8(associated+np.float32(.5)*(1-analysis_a[:, :, None]))
    sub = original[y0:y1, x0:x1]
    comp = composite[y0:y1, x0:x1].astype(np.float32)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    ex, ey = params['ellipse_radius']
    region = _smooth((1-((xs-cx)/ex)**2-((ys-cy)/ey)**2)/.4)
    dark = ((sub.max(2) < 43) & (np.abs(sub[:, :, 0]-sub[:, :, 1]) < 10) &
            (matte[y0:y1, x0:x1] > 232))
    if not dark.any():
        return channels
    near = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 5)
    cblue = comp[:, :, 2]-comp[:, :, 0]
    cgreen = comp[:, :, 1]-comp[:, :, 0]
    seed = ((cblue > 8) & (cgreen > 3) & (sub[:, :, 2]-sub[:, :, 0] > 11) &
            (near < 14) & (region > .02))
    count, ids, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8))
    strong = np.zeros(seed.shape, bool)
    for component in range(1, count):
        island = ids == component
        if stats[component, cv2.CC_STAT_AREA] >= 3 and cblue[island].max() >= 12:
            strong |= island
    if not strong.any():
        return channels
    distance = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 5)
    zone = (region*_smooth((12-distance)/7)*_smooth((19-near)/7)).astype(np.float32)
    sy = params['sky_sample_half_height']
    sample = original[max(0, int(cy)-sy):int(cy)+sy+1]
    sample_a = matte[max(0, int(cy)-sy):int(cy)+sy+1]
    dx = np.abs(np.arange(aw)-cx)
    sky_mask = ((sample_a < 1) & (sample[:, :, 2]-sample[:, :, 0] > 27) &
                (sample[:, :, 1]-sample[:, :, 0] > 15) &
                (dx[None, :] < params['sky_sample_half_width']))
    hair_samples = sub[dark & (region > .1)]
    if sky_mask.sum() <= 24 or len(hair_samples) < 20:
        return channels
    sky = np.median(sample[sky_mask], axis=0)/255
    hair = np.percentile(hair_samples, 30, axis=0).astype(np.float32)/255
    vector = sky-hair
    denominator = vector@vector
    if denominator <= np.finfo(np.float32).eps:
        return channels
    left, right = x0*width//aw, x1*width//aw
    top, bottom = y0*height//ah, y1*height//ah
    f = foreground[top:bottom, left:right]
    a = alpha[top:bottom, left:right]
    fraction = np.clip(((f-hair)*vector).sum(2)/denominator, 0, 1)
    residual = f-(hair+fraction[:, :, None]*vector)
    fit = _smooth((.175-np.max(np.abs(residual), axis=2))/.09)
    fblue = (f[:, :, 2]-f[:, :, 0])*255
    fgreen = (f[:, :, 1]-f[:, :, 0])*255
    screen = _smooth((fblue-3)/10)*_smooth((fgreen+1)/7)
    mask = cv2.resize(zone, (right-left, bottom-top),
                      interpolation=cv2.INTER_LINEAR)*fit*screen
    removal = a*fraction*mask
    changed = removal > 1e-7
    if not changed.any():
        return channels
    new_a = a-removal
    old_pm = f*a[:, :, None]
    # Mix associated pairs, then unassociate once. Reducing A lets a future
    # replacement background show through; simply desaturating F would not.
    new_pm = (old_pm*(1-mask[:, :, None]) +
              hair*(a*(1-fraction)*mask)[:, :, None])
    new_f = f.copy()
    new_f[changed] = new_pm[changed]/np.maximum(new_a[changed, None], 1e-8)
    result = {c: values.copy() for c, values in channels.items()}
    for i, c in enumerate('RGB'):
        result[c][top:bottom, left:right][changed] = new_f[:, :, i][changed]
    result['A'][top:bottom, left:right][changed] = new_a[changed]
    return result
