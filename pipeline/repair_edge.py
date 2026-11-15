"""CPU, native-float jacket-edge correction; no I/O, inference or rendering.

This is a shot-calibrated repair for an opaque textile against a distinct wall.
It is deliberately not a general matte feather. The source RGB locates the edge;
two *original* neighboring F32 mattes provide its position seed and softness.
Only the configured lower-shoulder strip is changed. RGB stays unassociated.
"""

import math

import cv2
import numpy as np


def _pair(value, name, *, minimum=None, inclusive=False):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int for v in value)
            or value[0] > value[1] or not inclusive and value[0] == value[1]
            or minimum is not None and value[0] < minimum):
        raise ValueError(f'{name} must be an ordered pair of integer bounds')
    return tuple(value)


def _positive(value, name, *, odd=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    if odd and (type(value) is not int or value % 2 != 1):
        raise ValueError(f'{name} must be a positive odd integer')
    return value


def _f32(plane, shape, name, *, coverage=False):
    if (not isinstance(plane, np.ndarray) or plane.dtype != np.float32
            or plane.ndim != 2 or plane.shape != shape
            or not np.isfinite(plane).all()):
        raise ValueError(f'{name} must be finite native float32 on the RGBA grid')
    if coverage and (plane.min() < 0 or plane.max() > 1):
        raise ValueError(f'{name} coverage must be within 0..1')


def _single_crossing(values, threshold, name, *, strict=False):
    """Find the sole increasing threshold crossing, rejecting guessed edges."""
    above = values > threshold if strict else values >= threshold
    rising = np.sum(~above[:, :-1] & above[:, 1:], axis=1)
    falling = np.sum(above[:, :-1] & ~above[:, 1:], axis=1)
    if np.any(above[:, 0]) or np.any(~above[:, -1]) or np.any(rising != 1) or np.any(falling):
        raise ValueError(f'Invalid or ambiguous {name}: require one rising edge per row')
    return np.argmax(above, axis=1)


def apply_edge(channels, rgb, local, rule, reference_alpha):
    """Return corrected native straight R/G/B/A planes, leaving inputs untouched.

    ``local`` and all rule frame indices are zero-based within the producer.
    ``frames`` is inclusive; coordinate windows with ``_columns`` or
    ``search_shifts`` are half-open, while sample/crossing shift pairs include
    both ends. ``reference_alpha(index)`` must return the producer's original
    native F32 alpha, never an earlier repaired or quantized result. Calibration
    previews are derived in memory using INTER_AREA and PNG8 rounding, exactly
    matching the historical repair's measured reference profiles.

    Invalid geometry, zero color contrast, missing edges and multiple crossings
    raise ValueError instead of silently selecting a silhouette. No pixels are
    resized in the returned channels, and no RGB is multiplied by alpha.
    """
    if not isinstance(rule, dict) or rule.get('type') != 'jacket-edge':
        raise ValueError('Expected a jacket-edge rule')
    frames = _pair(rule.get('frames'), 'frames', minimum=0, inclusive=True)
    if type(local) is not int or local < 0:
        raise ValueError('local must be a nonnegative integer frame index')
    if not isinstance(channels, dict) or set(channels) != set('RGBA'):
        raise ValueError('Expected straight R/G/B/A channels')
    shape = getattr(channels['A'], 'shape', None)
    if shape is None or len(shape) != 2 or min(shape) < 2:
        raise ValueError('Expected a two-dimensional native RGBA grid')
    for name in 'RGBA':
        _f32(channels[name], shape, name, coverage=name == 'A')
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError('Source must be same-frame native uint8 RGB on the RGBA grid')
    if not frames[0] <= local <= frames[1]:
        return {c: channels[c].copy() for c in 'RGBA'}

    required = {
        'type', 'frames', 'reference_frames', 'calibration_grid',
        'reference_encoding', 'edge_columns', 'first_row', 'center_threshold',
        'width_thresholds', 'search_shifts', 'wall_sample_shifts',
        'fabric_sample_shifts', 'crossing_shifts', 'profile_kernel',
        'profile_sigma', 'center_blur_rows', 'width_blur_rows', 'logistic_span',
        'minimum_sigma', 'coverage_snap', 'vertical_ramp', 'left_ramp_start',
        'left_ramp_width', 'right_ramp_end', 'right_ramp_width',
        'fabric_blur_rows', 'texture_offset', 'texture_ramp', 'color_end',
        'color_ramp', 'minimum_color_contrast_squared',
    }
    if set(rule) != required:
        raise ValueError(f'Jacket-edge rule keys differ: missing={sorted(required-set(rule))}, '
                         f'unknown={sorted(set(rule)-required)}')
    refs = _pair(rule['reference_frames'], 'reference_frames', minimum=0)
    if not refs[0] < frames[0] <= frames[1] < refs[1]:
        raise ValueError('Original reference_frames must bracket the repaired frames')
    grid = rule['calibration_grid']
    if (not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(type(v) is not int or v < 2 for v in grid)):
        raise ValueError('calibration_grid must contain positive width and height')
    cw, ch = grid
    if rule['reference_encoding'] != 'area-png8':
        raise ValueError('Only the calibrated area-png8 reference encoding is supported')
    col0, col1 = _pair(rule['edge_columns'], 'edge_columns', minimum=0)
    if col1 > cw:
        raise ValueError('edge_columns exceed calibration grid')
    first_row = rule['first_row']
    if type(first_row) is not int or not 1 <= first_row < ch - 1:
        raise ValueError('first_row must leave at least two calibration rows')
    low, high = _pair(rule['width_thresholds'], 'width_thresholds', minimum=1)
    center_threshold = rule['center_threshold']
    if type(center_threshold) is not int or not low < center_threshold < high <= 254:
        raise ValueError('Expected 0 < low < center_threshold < high < 255')
    start, stop = _pair(rule['search_shifts'], 'search_shifts')
    shift = np.arange(start, stop)
    masks = {}
    for key in ('wall_sample_shifts', 'fabric_sample_shifts', 'crossing_shifts'):
        a, b = _pair(rule[key], key)
        if a < start or b >= stop:
            raise ValueError(f'{key} exceed search_shifts')
        masks[key] = (shift >= a) & (shift <= b)
    wall_bounds = rule['wall_sample_shifts']
    fabric_bounds = rule['fabric_sample_shifts']
    crossing_bounds = rule['crossing_shifts']
    # The measured search window can overlap the inner cloth sample by a few
    # pixels; the historical calibration does so at shifts 34 and 35.
    if not (wall_bounds[1] < crossing_bounds[0] < crossing_bounds[1] < fabric_bounds[1]
            and wall_bounds[1] < fabric_bounds[0]):
        raise ValueError('Wall, crossing and fabric sample windows have invalid ordering')
    kernel = rule['profile_kernel']
    if not isinstance(kernel, (list, tuple)) or len(kernel) != 2:
        raise ValueError('profile_kernel must have two odd integer dimensions')
    for value in kernel:
        _positive(value, 'profile_kernel', odd=True)
    for key in ('center_blur_rows', 'width_blur_rows', 'fabric_blur_rows'):
        _positive(rule[key], key, odd=True)
    for key in ('profile_sigma', 'logistic_span', 'minimum_sigma',
                'vertical_ramp', 'left_ramp_width', 'right_ramp_width',
                'texture_ramp', 'color_ramp', 'minimum_color_contrast_squared'):
        _positive(rule[key], key)
    snap = rule['coverage_snap']
    if isinstance(snap, bool) or not isinstance(snap, (float, int)) or not 0 <= snap < .5:
        raise ValueError('coverage_snap must be finite within [0, .5)')
    for key in ('left_ramp_start', 'right_ramp_end', 'texture_offset', 'color_end'):
        if isinstance(rule[key], bool) or not isinstance(rule[key], (int, float)) or not math.isfinite(rule[key]):
            raise ValueError(f'{key} must be a finite number')
    if not start <= rule['left_ramp_start'] < rule['right_ramp_end'] < stop:
        raise ValueError('Spatial ramps must be inside the search strip')
    if kernel[0] > len(shift) or kernel[1] > shape[0]:
        raise ValueError('profile_kernel exceeds the native sample strip')

    h, w = shape
    first = int(first_row * h / ch)
    rows = np.arange(first, h)
    if len(rows) < 2:
        raise ValueError('Native repair region has fewer than two rows')
    for_ref = (np.arange(ch) + .5) * h / ch - .5
    # The first native row can interpolate from the calibration row above ROI.
    seed_first = max(0, int(np.searchsorted(for_ref, first, side='right')) - 1)
    ref_edges, widths = [], []
    for nearby in refs:
        ref = reference_alpha(nearby)
        _f32(ref, shape, f'original reference {nearby}', coverage=True)
        preview = np.floor(np.clip(cv2.resize(ref, (cw, ch), interpolation=cv2.INTER_AREA), 0, 1) * 255 + .5).astype(np.uint8)
        strip = preview[seed_first:, col0:col1]
        edge = _single_crossing(strip, center_threshold, 'reference center', strict=True) + col0
        ref_edges.append(np.interp(rows, for_ref[seed_first:], (edge + .5) * w / cw - .5))
        strip = preview[first_row:, col0:col1]
        lower = _single_crossing(strip, low, 'reference lower coverage')
        upper = _single_crossing(strip, high, 'reference upper coverage')
        measured = upper - lower
        if np.any(measured <= 0):
            raise ValueError('Reference edge has no measurable transition width')
        widths.append(np.interp(rows, for_ref[first_row:], measured.astype(float)) * w / cw)
    t = (local - refs[0]) / (refs[1] - refs[0])
    center = np.rint(ref_edges[0] + t * (ref_edges[1] - ref_edges[0])).astype(int)
    x = center[:, None] + shift
    if x.min() < 0 or x.max() >= w:
        raise ValueError('Native jacket-edge sample strip exceeds the image bounds')
    photo = rgb[rows[:, None], x].astype(np.float32) / 255
    wall = np.median(photo[:, masks['wall_sample_shifts']], axis=1)
    fabric = np.median(photo[:, masks['fabric_sample_shifts']], axis=1)
    vec = fabric - wall
    norm = np.sum(vec * vec, axis=1)
    if np.any(norm <= rule['minimum_color_contrast_squared']):
        raise ValueError('Wall and fabric colors have insufficient contrast')
    measured_coverage = ((photo - wall[:, None]) * vec[:, None]).sum(2) / norm[:, None]
    profile = cv2.GaussianBlur(measured_coverage.astype(np.float32), tuple(kernel), rule['profile_sigma'])
    cut = masks['crossing_shifts']
    crossing_profile = profile[:, cut]
    mid = _single_crossing(crossing_profile, .5, 'source-color crossing')
    if np.any(mid >= crossing_profile.shape[1] - 1):
        raise ValueError('Source-color edge touches the crossing window boundary')
    rr = np.arange(len(rows))
    right, left = crossing_profile[rr, mid], crossing_profile[rr, mid - 1]
    centers = center + shift[cut][mid - 1] + np.clip((.5 - left) / np.maximum(right - left, .001), 0, 1)
    centers = cv2.GaussianBlur(centers[:, None], (1, rule['center_blur_rows']), 0).ravel()
    sigma = cv2.GaussianBlur(((widths[0] + widths[1]) / 2)[:, None], (1, rule['width_blur_rows']), 0).ravel() / rule['logistic_span']
    coverage = (1 / (1 + np.exp(-np.clip((x - centers[:, None]) / np.maximum(sigma[:, None], rule['minimum_sigma']), -40, 40)))).astype(np.float32)
    coverage[coverage < snap] = 0
    coverage[coverage > 1 - snap] = 1
    wy = np.clip((rows - first) / rule['vertical_ramp'], 0, 1)[:, None]
    wx = (np.clip((shift - rule['left_ramp_start']) / rule['left_ramp_width'], 0, 1)
          * np.clip((rule['right_ramp_end'] - shift) / rule['right_ramp_width'], 0, 1))
    weight = (wy * wx).astype(np.float32)
    current_a = channels['A'][rows[:, None], x]
    new_a = current_a * (1 - weight) + coverage * weight
    inside = cv2.GaussianBlur(fabric, (1, rule['fabric_blur_rows']), 0)
    blend = np.clip((x - centers[:, None] - rule['texture_offset']) / rule['texture_ramp'], 0, 1).astype(np.float32)
    foreground = inside[:, None] * (1 - blend[:, :, None]) + photo * blend[:, :, None]
    color_weight = weight * np.clip((rule['color_end'] - shift) / rule['color_ramp'], 0, 1)[None, :]
    result = {c: channels[c].copy() for c in 'RGBA'}
    color_rows, color_cols = np.nonzero(color_weight > 0)
    for i, name in enumerate('RGB'):
        prev = channels[name][rows[:, None], x]
        new_f = prev * (1 - color_weight) + foreground[:, :, i] * color_weight
        result[name][rows[color_rows], x[color_rows, color_cols]] = new_f[color_rows, color_cols]
    alpha_rows, alpha_cols = np.nonzero(weight > 0)
    result['A'][rows[alpha_rows], x[alpha_rows, alpha_cols]] = new_a[alpha_rows, alpha_cols]
    return result
