import tempfile
import unittest
from pathlib import Path
import numpy as np
from pipeline.export import straight_channels, export_rgba
from pipeline.media import white_alpha, write_exr, read_exr, bit_exact
from pipeline.common import save_json, sha


class ExportTests(unittest.TestCase):
    def test_repair_clip_mismatch_leaves_no_partial_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root / 'frame-ledger.json', {
                'width': 4, 'height': 2, 'fps': 25, 'frame_count': 1,
                'frames': [{'source_frame_index': 75, 'inputs': {'source': {'sha256': 'a' * 64}}}]})
            recipe = root / 'recipe.json'
            save_json(recipe, {'grid': [4, 2], 'source_start_frame': 75,
                               'first_source_frame_sha256': 'b' * 64,
                               'rules': [{'type': 'hair-sky', 'frames': [0, 0]}]})
            with self.assertRaisesRegex(ValueError, 'different source clip'):
                export_rgba(root, root / 'corrected', repairs=recipe)
            self.assertFalse((root / 'corrected').exists())

    def test_straight_does_not_square_alpha_or_premultiply_RGB(self):
        f = {c: np.full((2,3), .8, np.float32) for c in 'RGB'}
        a = {c: np.ones((2,3), np.float32) for c in 'RGB'}
        a['A'] = np.full((2,3), .25, np.float32)
        rgba = straight_channels(f, a)
        self.assertTrue(bit_exact(rgba['R'], f['R']))
        self.assertTrue(bit_exact(rgba['A'], a['A']))
        a['R'][0,0] = .5
        with self.assertRaises(ValueError): straight_channels(f, a)

    def test_actual_F32_roundtrip_and_input_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); producer=root/'producer'; producer.mkdir()
            values=np.array([[0,.12345679,.5],[1e-8,.99999994,1]],np.float32)
            white_alpha(producer/'a.exr',values)
            write_exr(producer/'f.exr',{c:values.copy() for c in 'RGB'},'BT709-code-values')
            row={'local_index':0,'source_frame_index':75,'time_seconds':0.,'inputs':{
                'alpha':{'path':'a.exr','sha256':sha(producer/'a.exr')},
                'foreground':{'path':'f.exr','sha256':sha(producer/'f.exr')}}}
            save_json(producer/'frame-ledger.json',{'fps':25,'width':3,'height':2,'frame_count':1,'frames':[row]})
            export_rgba(producer,root/'out')
            actual=read_exr(root/'out/rgba_00001.exr')
            for c in 'RGBA': self.assertTrue(bit_exact(actual[c],values))
            export_rgba(producer,root/'preview','png8')
            with self.assertRaises(FileExistsError): export_rgba(producer,root/'out')
            with (producer/'f.exr').open('ab') as stream: stream.write(b'tamper')
            with self.assertRaises(ValueError): export_rgba(producer,root/'refused')
