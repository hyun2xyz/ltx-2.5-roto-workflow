"""CPU regressions; original pixels are optional external test inputs only.

To compare an archived native patch, set LTX_HAIR_PARITY_JSON to a JSON file
with source_rgb, base_rgba, native_patch and local keys. Paths resolve relative
to that file. Its recipe key may override the published hair-sky preset.
"""
import json
import os
import unittest
from pathlib import Path

import numpy as np

from pipeline.media import bit_exact, read_exr
from pipeline.repair_hair import apply_hair


def hair_gap():
    """Known straight hair/sky mixture next to opaque dark hair."""
    sky = np.array([149, 185, 206], np.float32)/255
    hair = np.array([2, 2, 7], np.float32)/255
    rgb = np.empty((180, 240, 3), np.uint8)
    rgb[:] = [149, 185, 206]
    rgb[65:115, 108:132] = [2, 2, 7]
    foreground = rgb.astype(np.float32)/255
    alpha = np.zeros((180, 240), np.float32)
    alpha[65:115, 108:132] = 1
    gap = np.s_[85:97, 116:124]
    fraction = np.float32(.65)
    foreground[gap] = hair+fraction*(sky-hair)
    rgb[gap] = np.floor(foreground[gap]*255+.5).astype(np.uint8)
    alpha[gap] = .95
    channels = {c: foreground[:, :, i].copy() for i, c in enumerate('RGB')}
    channels['A'] = alpha
    # Signed zero is valid FLOAT32 and must survive untouched exterior pixels.
    channels['R'][0, 0] = -0.
    channels['A'][0, 0] = -0.
    rule = {'type': 'hair-sky', 'frames': [0, 2],
            'analysis_size': [240, 180], 'anchors': [[0, 120, 90], [2, 120, 90]]}
    return channels, rgb, rule, hair, sky, gap, fraction


class HairRepairTests(unittest.TestCase):
    def test_known_mixture_reduces_coverage_and_returns_straight_hair(self):
        channels, rgb, rule, hair, sky, gap, fraction = hair_gap()
        before = {c: values.copy() for c, values in channels.items()}
        original_rgb = rgb.copy()
        actual = apply_hair(channels, rgb, 1, rule)
        self.assertIsNot(actual, channels)
        np.testing.assert_allclose(actual['A'][gap], .95*(1-fraction), atol=2e-7)
        for i, c in enumerate('RGB'):
            np.testing.assert_allclose(actual[c][gap], hair[i], atol=1e-7)
        changed = np.zeros(channels['A'].shape, bool)
        for c in 'RGBA':
            self.assertTrue(bit_exact(channels[c], before[c]), 'Mutated input '+c)
            self.assertEqual(actual[c].dtype, np.float32)
            self.assertTrue(np.isfinite(actual[c]).all())
            changed |= actual[c].view(np.uint32) != before[c].view(np.uint32)
        np.testing.assert_array_equal(rgb, original_rgb)
        self.assertGreater(changed.sum(), 0)
        outside = np.ones(changed.shape, bool)
        outside[6:174, 29:211] = False
        for c in 'RGBA':
            self.assertTrue(bit_exact(actual[c][outside], before[c][outside]))
        self.assertTrue(np.all(actual['A'] <= before['A']))
        # The pair changes, but recompositing onto the estimated original sky
        # reconstructs the same exposure. A desaturation-only fix fails this.
        old_f = np.stack([before[c] for c in 'RGB'], axis=2)
        new_f = np.stack([actual[c] for c in 'RGB'], axis=2)
        old_comp = old_f*before['A'][:, :, None]+sky*(1-before['A'][:, :, None])
        new_comp = new_f*actual['A'][:, :, None]+sky*(1-actual['A'][:, :, None])
        np.testing.assert_allclose(new_comp, old_comp, atol=2e-7)

    def test_insufficient_evidence_returns_the_same_channels(self):
        channels, rgb, rule, *_ = hair_gap()
        rgb[:] = [149, 185, 206]  # No neighboring dark hair.
        self.assertIs(apply_hair(channels, rgb, 0, rule), channels)
        channels, rgb, rule, *_ = hair_gap()
        channels['A'][:] = 1  # No unoccluded sky sample.
        self.assertIs(apply_hair(channels, rgb, 0, rule), channels)

    def test_invalid_precision_grid_and_track_are_rejected(self):
        channels, rgb, rule, *_ = hair_gap()
        with self.assertRaises(ValueError):
            apply_hair({**channels, 'R': channels['R'].astype(np.float16)}, rgb, 0, rule)
        with self.assertRaises(ValueError):
            apply_hair(channels, rgb[:-1], 0, rule)
        with self.assertRaises(ValueError):
            apply_hair(channels, rgb, 0, {**rule, 'anchors': [[1, 120, 90], [1, 121, 90]]})
        with self.assertRaises(ValueError):
            apply_hair(channels, rgb, 0, {**rule, 'params': {'unknown': 1}})

    @unittest.skipUnless(os.environ.get('LTX_HAIR_PARITY_JSON'), 'No external native patch supplied')
    def test_archived_native_patch_is_bit_exact_and_exterior_is_unchanged(self):
        import OpenImageIO as oiio
        from PIL import Image
        manifest = Path(os.environ['LTX_HAIR_PARITY_JSON']).resolve(strict=True)
        case = json.loads(manifest.read_text(encoding='utf-8-sig'))
        root = manifest.parent
        preset = Path(__file__).resolve().parents[1]/'config/repairs/hair-sky.json'
        recipe = json.loads((root/Path(case.get('recipe', preset))).read_text(encoding='utf-8-sig'))
        channels = read_exr(root/Path(case['base_rgba']))
        with Image.open(root/Path(case['source_rgb'])) as image:
            rgb = np.asarray(image.convert('RGB'))
        actual = apply_hair(channels, rgb, case['local'], recipe['rules'][0])
        handle = oiio.ImageInput.open(str(root/Path(case['native_patch'])))
        self.assertIsNotNone(handle)
        try:
            spec = handle.spec()
            x, y, w, h = spec.x, spec.y, spec.width, spec.height
            patch = handle.read_image(format=oiio.FLOAT)
            names = list(spec.channelnames)
            self.assertEqual((spec.full_width, spec.full_height), channels['A'].shape[::-1])
        finally:
            handle.close()
        outside = np.ones(channels['A'].shape, bool)
        outside[y:y+h, x:x+w] = False
        for i, c in enumerate(names):
            self.assertTrue(bit_exact(actual[c][y:y+h, x:x+w], patch[:, :, i]), c)
            self.assertTrue(bit_exact(actual[c][outside], channels[c][outside]), c)
