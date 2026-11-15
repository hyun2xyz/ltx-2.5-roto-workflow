import copy
import json
import unittest
from pathlib import Path

import numpy as np

from pipeline.repair_edge import apply_edge


PRESET = Path(__file__).resolve().parents[1] / 'config/repairs/jacket-edge.json'


def fixture(local=10):
    """A misplaced, over-soft alpha beside a clearly measured opaque jacket."""
    h, w = 64, 160
    x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))

    def logistic(center, sigma):
        return (1 / (1 + np.exp(-(x - center) / sigma))).astype(np.float32)

    reference = {9: logistic(64, 2), 12: logistic(67, 2)}
    measured = logistic(65.35 + local - 10, 2)
    wall = np.array([.8, .78, .75], np.float32)
    cloth = np.array([.15, .19, .24], np.float32)
    photo = wall + measured[:, :, None] * (cloth - wall)
    rgb = np.floor(photo * 255 + .5).astype(np.uint8)
    channels = {c: np.full((h, w), v, np.float32) for c, v in zip('RGB', (.6, .5, .4))}
    channels['A'] = logistic(61, 5)
    rule = json.loads(PRESET.read_text(encoding='utf-8'))['rules'][0]
    rule.update({
        'frames': [10, 11], 'reference_frames': [9, 12],
        'calibration_grid': [w, h], 'edge_columns': [35, 115],
        'first_row': 16, 'search_shifts': [-25, 26],
        'wall_sample_shifts': [-22, -12], 'fabric_sample_shifts': [12, 20],
        'crossing_shifts': [-8, 8], 'vertical_ramp': 8,
        'left_ramp_start': -24, 'left_ramp_width': 4,
        'right_ramp_end': 24, 'right_ramp_width': 6,
        'texture_offset': 3, 'texture_ramp': 6, 'color_end': 14, 'color_ramp': 6,
    })
    return channels, rgb, rule, reference


class JacketEdgeTests(unittest.TestCase):
    def test_localized_native_correction_stays_straight_and_preserves_inputs(self):
        for local in (10, 11):
            with self.subTest(local=local):
                channels, rgb, rule, refs = fixture(local)
                # Zero-weight boundary pixels must retain their exact F32 bits,
                # including signed zero (arithmetic by zero can change its sign).
                channels['R'][16] = np.float32(-0.)
                originals = {c: v.copy() for c, v in channels.items()}
                reference_originals = {n: v.copy() for n, v in refs.items()}
                calls = []

                def reference(n):
                    calls.append(n)
                    return refs[n]

                result = apply_edge(channels, rgb, local, rule, reference)
                self.assertEqual(calls, [9, 12])
                for c in 'RGBA':
                    self.assertEqual(result[c].dtype, np.float32)
                    self.assertEqual(result[c].shape, channels[c].shape)
                    self.assertFalse(np.shares_memory(result[c], channels[c]))
                    np.testing.assert_array_equal(channels[c], originals[c])
                    np.testing.assert_array_equal(result[c][:16], channels[c][:16])
                    np.testing.assert_array_equal(result[c][16].view(np.uint32), channels[c][16].view(np.uint32))
                    np.testing.assert_array_equal(result[c][:, :39], channels[c][:, :39])
                    np.testing.assert_array_equal(result[c][:, 92:], channels[c][:, 92:])
                for n in refs:
                    np.testing.assert_array_equal(refs[n], reference_originals[n])
                edge = 65 + local - 10
                # The source photo, rather than the defective input alpha at61,
                # determines the corrected edge center. Neighboring clean
                # mattes constrain the transition to about nine native pixels.
                self.assertLess(result['A'][40, edge], .5)
                self.assertGreater(result['A'][40, edge + 1], .5)
                band = np.flatnonzero((result['A'][40] >= .1) & (result['A'][40] <= .9))
                self.assertLessEqual(len(band), 10)
                self.assertGreater(np.max(np.abs(result['A'] - channels['A'])), .25)
                # Recovered cloth retains its straight color even at low alpha.
                self.assertLess(result['A'][40, edge - 4], .2)
                for c, value in zip('RGB', (.15, .19, .24)):
                    self.assertAlmostEqual(float(result[c][40, edge - 4]), value, delta=.003)

    def test_historical_overlap_and_single_frame_inclusive_selection(self):
        channels, rgb, rule, refs = fixture()
        rule['frames'] = [10, 10]
        rule['crossing_shifts'] = [-8, 14]  # Overlaps the inner cloth sample.
        corrected = apply_edge(channels, rgb, 10, rule, refs.__getitem__)
        self.assertFalse(np.array_equal(corrected['A'], channels['A']))
        unchanged = apply_edge(channels, rgb, 11, rule,
                               lambda _: self.fail('Unselected frame read a reference'))
        for c in 'RGBA':
            np.testing.assert_array_equal(unchanged[c], channels[c])

    def test_rejects_ambiguous_edge_and_insufficient_color_contrast(self):
        channels, rgb, rule, refs = fixture()
        with self.assertRaisesRegex(ValueError, 'insufficient contrast'):
            apply_edge(channels, np.full_like(rgb, 180), 10, rule, refs.__getitem__)
        ambiguous = {n: alpha.copy() for n, alpha in refs.items()}
        for alpha in ambiguous.values():
            alpha[:, 48:52] = 1
        with self.assertRaisesRegex(ValueError, 'ambiguous reference center'):
            apply_edge(channels, rgb, 10, rule, ambiguous.__getitem__)
        # Real photographic detail can create several source-color crossings.
        # Do not silently take the first crossing as the original argmax did.
        raw = rgb.copy()
        raw[:, 61:65] = np.array([38, 48, 61], np.uint8)
        raw[:, 65:69] = np.array([204, 199, 191], np.uint8)
        with self.assertRaisesRegex(ValueError, 'ambiguous source-color crossing'):
            apply_edge(channels, raw, 10, rule, refs.__getitem__)

    def test_rejects_invalid_dtype_grid_geometry_and_config(self):
        channels, rgb, rule, refs = fixture()
        bad = {**channels, 'A': channels['A'].astype(np.float16)}
        with self.assertRaisesRegex(ValueError, 'float32'):
            apply_edge(bad, rgb, 10, rule, refs.__getitem__)
        with self.assertRaisesRegex(ValueError, 'uint8 RGB'):
            apply_edge(channels, rgb[:, :-1], 10, rule, refs.__getitem__)
        with self.assertRaisesRegex(ValueError, 'original reference'):
            apply_edge(channels, rgb, 10, rule, lambda n: refs[n].astype(np.float64))
        for key, value in (
            ('reference_frames', [10, 12]),
            ('edge_columns', [35, 161]),
            ('first_row', 63),
            ('profile_kernel', [4, 9]),
            ('profile_sigma', float('nan')),
            ('wall_sample_shifts', [-40, -12]),
            ('minimum_color_contrast_squared', 0),
            ('reference_encoding', 'native-half'),
        ):
            with self.subTest(key=key):
                bad_rule = {**rule, key: value}
                with self.assertRaises(ValueError):
                    apply_edge(channels, rgb, 10, bad_rule, refs.__getitem__)
        bad_rule = copy.deepcopy(rule)
        bad_rule['search_shifts'] = [-80, 26]
        with self.assertRaisesRegex(ValueError, 'image bounds'):
            apply_edge(channels, rgb, 10, bad_rule, refs.__getitem__)

    def test_shot_specific_preset_has_source_clock_and_portable_parameters(self):
        preset = json.loads(PRESET.read_text(encoding='utf-8'))
        self.assertEqual(preset['source_start_frame'], 75)
        self.assertEqual(preset['grid'], [4096, 2160])
        self.assertEqual(len(preset['first_source_frame_sha256']), 64)
        self.assertEqual(preset['rules'][0]['frames'], [56, 57])
        self.assertEqual(preset['rules'][0]['reference_frames'], [55, 58])
        self.assertNotIn('path', PRESET.read_text(encoding='utf-8').lower())


if __name__ == '__main__':
    unittest.main()
