import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.common import frame_contract, read_json, save_json
from pipeline.local_api import LocalAPI, status, submit

class FakeAPI:
    def __init__(self, uncertain=False, mismatch=False):
        self.posts=[];self.gets=[];self.uncertain=uncertain;self.mismatch=mismatch
    def request(self,path,payload=None):
        if payload is not None:
            self.posts.append((path,payload))
            if self.uncertain:raise TimeoutError('Reply lost after possible acceptance')
            return {'prompt_id':'unexpected-server-id' if self.mismatch else payload['prompt_id']}
        self.gets.append(path)
        return {'queue_running':[],'queue_pending':[]} if path=='/queue' else {}

class SubmissionTests(unittest.TestCase):
    def make_run(self,path):
        save_json(path/'state.json',{'phase':'prepared'})
        save_json(path/'prompt_api.json',{'1':{'class_type':'Example','inputs':{}}})
    def test_repeated_submit_and_known_uuid_never_post_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);self.make_run(run);api=FakeAPI()
            first=submit(run,api);submit(run,api);status(run,api)
            self.assertEqual(len(api.posts),1)
            self.assertEqual(first['prompt_id'],read_json(run/'submission-intent.json')['prompt_id'])
            self.assertEqual(read_json(run/'submission-response.json')['prompt_id'],first['prompt_id'])
    def test_uncertain_reply_resumes_only_by_get(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);self.make_run(run);api=FakeAPI(uncertain=True)
            with self.assertRaises(TimeoutError):submit(run,api)
            self.assertEqual(read_json(run/'state.json')['phase'],'submission_uncertain')
            submit(run,api);self.assertEqual(len(api.posts),1)
    def test_crash_after_intent_before_state_never_posts(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);self.make_run(run)
            save_json(run/'submission-intent.json',{'prompt_id':'persisted-owned-uuid','client_id':'client'})
            api=FakeAPI();result=submit(run,api)
            self.assertEqual(result['prompt_id'],'persisted-owned-uuid');self.assertEqual(api.posts,[])
    def test_mismatched_response_is_saved_and_never_reposted(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);self.make_run(run);api=FakeAPI(mismatch=True)
            with self.assertRaises(RuntimeError):submit(run,api)
            self.assertEqual(read_json(run/'submission-response.json')['prompt_id'],'unexpected-server-id')
            submit(run,api);self.assertEqual(len(api.posts),1)
    def test_cloud_and_credential_urls_refused(self):
        for url in ['https://example.com','http://192.0.2.1:8197','http://u:p@127.0.0.1:8197','http://127.0.0.1:8197/api']:
            with self.assertRaises(ValueError):LocalAPI(url)
        self.assertEqual(LocalAPI('http://localhost:8197').url,'http://127.0.0.1:8197')

class ClockAndWorkflowTests(unittest.TestCase):
    def test_65frame_numbering_and_clock(self):
        rows=[frame_contract(i) for i in range(65)]
        self.assertEqual(rows[0],{'local_index':0,'source_frame_index':75,'file_index':1,'time_seconds':0.})
        self.assertEqual(rows[-1]['source_frame_index'],139);self.assertEqual(rows[-1]['file_index'],65)
        self.assertEqual(rows[-1]['time_seconds'],2.56);self.assertEqual(len(rows)/25,2.6)
        self.assertTrue(all(abs(rows[i+1]['time_seconds']-rows[i]['time_seconds']-.04)<1e-12 for i in range(64)))
    def test_actual_15node_template_and_precision_settings(self):
        from pipeline.common import REPO
        from pipeline.cli import replace_values
        prompt=replace_values(read_json(REPO/'workflows/ltx_fusion65_api.json'),{'__REFERENCE__':'/external/run/assets/reference.mp4'})
        self.assertEqual(len(prompt),15)
        self.assertEqual(prompt['5']['inputs']['video'],'/external/run/assets/reference.mp4')
        self.assertEqual(prompt['5']['inputs']['frame_load_cap'],65);self.assertEqual(prompt['5']['inputs']['skip_first_frames'],0)
        self.assertEqual(prompt['11']['inputs']['use_streaming'],True);self.assertFalse(prompt['11']['inputs']['use_tiled_encode'])
        self.assertEqual(prompt['16']['inputs']['tile_frames'],33);self.assertEqual(prompt['16']['inputs']['seed'],42)
        self.assertEqual([prompt['16']['inputs'][k] for k in ('tile_width','tile_height')],[1920,1088])
        self.assertEqual(prompt['14']['inputs']['sampler_name'],'euler')
        self.assertEqual(prompt['21']['inputs']['format.bit_depth'],'32-bit float')
        self.assertEqual(prompt['21']['inputs']['format.input_color_space'],'linear')

@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy is optional for stdlib-only submission tests')
class FloatContractTests(unittest.TestCase):
    def test_precision_rejects_half_and_preserves_low_bits(self):
        import numpy as np
        from pipeline.media import bit_exact, require_f32
        values=np.array([.12345679,np.nextafter(np.float32(.5),np.float32(1)),1e-8],np.float32)
        self.assertTrue(bit_exact(values,values.copy()))
        self.assertFalse(bit_exact(values,values.astype(np.float16).astype(np.float32)))
        with self.assertRaises(ValueError):require_f32(values.astype(np.float16))
    def test_positive_overlap_matches_production_geometry_hash(self):
        from pipeline.media import pixel_sha,tile_geometry
        xs,ys,weight,norm=tile_geometry()
        self.assertEqual(len(xs)*len(ys),28);self.assertEqual(ys[-1],1136)
        self.assertTrue((norm>0).all())
        self.assertEqual(pixel_sha(weight),'954fef57ba2c67a7dd8d6b6519fcbf93a799baec3e100e0d9e260637fcd93eaa')
        self.assertEqual(pixel_sha(norm),'764fe453f9fcd6f652747f374a58f162cfdd6363d8664be3d01bae4f300fc562')
    @unittest.skipUnless(importlib.util.find_spec('OpenImageIO'),'OpenImageIO required for actual EXR header regression')
    def test_mixed_half_float_EXR_is_rejected(self):
        import numpy as np
        import OpenImageIO as oiio
        from pipeline.media import read_exr
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mixed.exr';spec=oiio.ImageSpec(4,3,2,oiio.FLOAT)
            spec.channelnames=['R','A'];spec.alpha_channel=1;spec.channelformats=[oiio.FLOAT,oiio.HALF]
            handle=oiio.ImageOutput.create(str(path));self.assertTrue(handle.open(str(path),spec))
            try:self.assertTrue(handle.write_image(np.ones((3,4,2),np.float32)))
            finally:handle.close()
            with self.assertRaises(ValueError):read_exr(path)

if __name__=='__main__':unittest.main()
