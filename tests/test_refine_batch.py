"""Small orchestration tests; no GPU, model load, real ML call or native render."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pipeline import refine
from pipeline.common import frame_contract, read_json, save_json, sha
from pipeline.media import pixel_sha

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[]
        for local in (8,32):
            inputs={}
            for kind in ('source','raw_prior'):
                path=self.root/f'input-{kind}-{local}.bin';path.write_bytes(f'{kind}/{local}'.encode())
                inputs[kind]={'path':str(path),'sha256':sha(path)}
            self.rows.append({**frame_contract(local),'inputs':inputs})
        save_json(self.root/'plan.json',{'method':refine.METHOD,'config':{},'frames':self.rows})

    def fake_tile(self,out,row,cfg,context=None):
        self.assertIs(context,self.context)
        alpha=np.full((2,3),np.float32(.12345679),np.float32)
        np.save(out/'alpha.npy',alpha,allow_pickle=False)
        save_json(out/'inference.json',{'A_pixel_sha256':pixel_sha(alpha)})
        (out/'trimap.png').write_bytes(b'test-only-trimap')

    def fake_foreground(self,out,row):
        alpha=np.load(out/'alpha.npy',allow_pickle=False)
        foreground=np.full((*alpha.shape,3),np.float32(.31415927),np.float32)
        np.save(out/'foreground.npy',foreground,allow_pickle=False)
        save_json(out/'foreground.json',{'A_pixel_sha256':pixel_sha(alpha),'F_pixel_sha256':pixel_sha(foreground)})

    def fake_pack(self,root,out,row):
        inputs={}
        for kind in ('source','raw_prior','alpha','foreground'):
            path=root/f"test-packed/{kind}-{row['local_index']}.bin";path.parent.mkdir(exist_ok=True)
            path.write_bytes(Path(row['inputs'][kind]['path']).read_bytes() if kind in row['inputs'] else kind.encode())
            inputs[kind]={'path':str(path),'sha256':sha(path)}
        save_json(root/f"receipts/frame_{row['local_index']+1:05d}.json",{
            **{k:row[k] for k in ('local_index','source_frame_index','file_index','time_seconds')},
            'method':refine.METHOD,'inputs':inputs})

    def tile(self):
        self.context={'setup_seconds':.1}
        with patch.object(refine,'load_tiled_context',return_value=self.context) as load,patch.object(refine,'tiled_frame',side_effect=self.fake_tile):
            result=refine.batch_worker('tile',self.root)
        return result,load.call_count

    def foreground(self):
        with patch.object(refine,'foreground_frame',side_effect=self.fake_foreground):
            return refine.batch_worker('foreground',self.root)

    def test_model_once_then_completed_frames_skip_all_work(self):
        result,calls=self.tile();self.assertEqual(calls,1);self.assertEqual(result['processed'],2)
        result,calls=self.tile();self.assertEqual(calls,0);self.assertEqual(result['reused'],2)
        self.foreground()
        with patch.object(refine,'pack_frame',side_effect=self.fake_pack):
            self.assertEqual(refine.batch_worker('pack',self.root)['processed'],2)
        with patch.object(refine,'load_tiled_context') as load,patch.object(refine,'foreground_frame') as fg,patch.object(refine,'pack_frame') as pack:
            for phase in ('tile','foreground','pack'):
                self.assertEqual(refine.batch_worker(phase,self.root)['reused'],2)
            load.assert_not_called();fg.assert_not_called();pack.assert_not_called()

    def test_changed_input_rejected_before_model_load(self):
        self.tile();Path(self.rows[0]['inputs']['source']['path']).write_bytes(b'changed')
        with patch.object(refine,'load_tiled_context') as load:
            with self.assertRaisesRegex(ValueError,'SHA changed'):refine.batch_worker('tile',self.root)
            load.assert_not_called()

    def test_changed_alpha_npy_rejected_before_foreground(self):
        self.tile();path=self.root/'working/local08/alpha.npy'
        values=np.load(path,allow_pickle=False);values[0,0]=np.float32(.9);np.save(path,values,allow_pickle=False)
        with patch.object(refine,'foreground_frame') as fg:
            with self.assertRaisesRegex(ValueError,'Intermediate file/data changed'):refine.batch_worker('foreground',self.root)
            fg.assert_not_called()

    def test_changed_foreground_npy_rejected_before_pack(self):
        self.tile();self.foreground();path=self.root/'working/local32/foreground.npy'
        values=np.load(path,allow_pickle=False);values[0,0,0]=np.float32(.9);np.save(path,values,allow_pickle=False)
        with patch.object(refine,'pack_frame') as pack:
            with self.assertRaisesRegex(ValueError,'Intermediate file/data changed'):refine.batch_worker('pack',self.root)
            pack.assert_not_called()

    def test_partial_outputs_and_concurrent_worker_are_not_overwritten(self):
        out=self.root/'working/local08';out.mkdir(parents=True);(out/'trimap.png').write_bytes(b'partial')
        with patch.object(refine,'load_tiled_context') as load:
            with self.assertRaisesRegex(RuntimeError,'Partial tile'):refine.batch_worker('tile',self.root)
            load.assert_not_called()
        self.assertEqual((out/'trimap.png').read_bytes(),b'partial')
        lock=self.root/'.batch-worker.lock';lock.write_text('existing-owner')
        with self.assertRaisesRegex(RuntimeError,'lock already exists'):refine.batch_worker('tile',self.root)
        self.assertEqual(lock.read_text(),'existing-owner')

    def test_completed_asset_change_is_not_reused(self):
        self.tile();self.foreground()
        with patch.object(refine,'pack_frame',side_effect=self.fake_pack):refine.batch_worker('pack',self.root)
        receipt=read_json(self.root/'receipts/frame_00009.json')
        Path(receipt['inputs']['alpha']['path']).write_bytes(b'changed-alpha')
        with patch.object(refine,'load_tiled_context') as load:
            with self.assertRaisesRegex(ValueError,'SHA changed'):refine.batch_worker('tile',self.root)
            load.assert_not_called()

if __name__=='__main__':unittest.main()
