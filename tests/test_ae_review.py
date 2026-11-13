"""Offline review5 guards. Fake files/host receipts do not certify an AE run."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ae'))
import queue_tools as q


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.delivery=self.root/'delivery'
        self.base=self.delivery/'attempts/review/LTX_BF16_4K_REFINED_ML_REVIEW.aep'
        self.base.parent.mkdir(parents=True);self.base.write_bytes(b'FAKE AEP: not a native project')
        self.settings_path=self.root/'settings.json'
        self.settings={'delivery_root':'delivery','producer_root':'producer','attempt':'review',
                       'output_template':'FAKE TEMPLATE','foreground_input_preserve_rgb_confirmed':True}
        self.save(self.settings_path,self.settings)
        self.a=np.array([[0,.25],[.5,1]],dtype=np.float32)
        self.f=np.full((2,2,3),.7,dtype=np.float32)
        self.rgba=np.dstack((self.f*self.a[:,:,None],self.a))
        self.matte=np.dstack((np.ones((2,2,3),dtype=np.float32),self.a))
        self.validation={'width':4096,'height':2160,'fps':25,'frame_count':65,
                         'full65_validated':True,'source_frame_range':[75,139],'frames':[]}
        for local in range(65):
            inputs={}
            for role in ('source','alpha','foreground','raw_prior'):
                path=self.root/'producer'/f'{role}-{local}.fake';path.parent.mkdir(exist_ok=True)
                path.write_bytes(f'FAKE INPUT {role} {local}'.encode())
                inputs[role]={'path':str(path),'sha256':q.sha(path)}
            self.validation['frames'].append({'local_index':local,'source_frame_index':75+local,
                                              'time_seconds':local/25,'inputs':inputs})
        self.validation_path=self.delivery/'input-validation.json';self.save(self.validation_path,self.validation)
        self.queue={'ok':True,'phase':'review','review_layer_clock_readback':True,
                    'foreground_input_preserve_rgb_confirmed':True,'base_project_sha256':q.sha(self.base),
                    'input_validation_sha256':q.sha(self.validation_path),'items':[]}
        for key in q.REVIEW_FRAMES:
            folder=self.root/'exports'/str(key);folder.mkdir(parents=True)
            (folder/'arbitrary_00000.exr').write_bytes(b'FAKE EXR: numeric decoder mocked')
            self.queue['items'].append({'branch':'clean','comp':'ROTO_CLEAN_REFINED_4K',
                'first_local_frame':key,'frame_count':1,'output_directory':str(folder),
                'render_settings':{'Time Span Start':str(key/25),'Time Span Duration':'0.04','Time Span End':str((key+1)/25)}})
        self.queue_path=self.root/'host.json';self.receipt=self.root/'proof.json'

    @staticmethod
    def save(path,value):
        path.write_text(json.dumps(value),encoding='utf-8')

    def verify(self,queue=None,header=None,pixels=None):
        self.save(self.queue_path,self.queue if queue is None else queue)
        header=header or {'width':4096,'height':2160,'compression_code':3,'channel_pixel_types':{c:2 for c in 'RGBA'}}
        def read(path,alpha=False):return (self.matte if alpha else self.f),{}
        def native(path):return (self.rgba if pixels is None else pixels),{'path':str(path),'sha256':q.sha(path),'header':header}
        with patch.dict(sys.modules,{'OpenImageIO':SimpleNamespace()}),patch.object(q,'read_exr',side_effect=read),patch.object(q,'native',side_effect=native),contextlib.redirect_stdout(io.StringIO()):
            q.verify_review(SimpleNamespace(settings=str(self.settings_path),queue_receipt=str(self.queue_path),receipt=str(self.receipt)))
        return json.loads(self.receipt.read_text())

    def test_review_reports_five_only_and_original_mapping(self):
        report=self.verify()
        self.assertEqual(report['status'],'PASS_NATIVE_REVIEW5_SAMPLES_ONLY')
        self.assertEqual(report['frames_checked'],5)
        self.assertFalse(report['all65_native_verified']);self.assertFalse(report['visual_approval'])
        self.assertFalse(report['source_RGB8_native_verified']);self.assertFalse(report['opaque_F_native_verified'])
        self.assertTrue(report['A_bit_exact_all5']);self.assertTrue(report['RGB_bit_exact_all5'])
        self.assertEqual([r['source_frame_index'] for r in report['frames']],[75,83,91,107,139])

    def test_missing_and_duplicate_keys_rejected(self):
        for change in ('missing','duplicate'):
            queue=copy.deepcopy(self.queue)
            if change=='missing':queue['items'].pop()
            else:queue['items'][1]['first_local_frame']=0
            with self.subTest(change=change),self.assertRaises(ValueError):self.verify(queue)

    def test_missing_extra_or_shared_output_rejected(self):
        folder=Path(self.queue['items'][0]['output_directory']);file=folder/'arbitrary_00000.exr'
        file.unlink()
        with self.assertRaises(ValueError):self.verify()
        file.write_bytes(b'fake');(folder/'extra.exr').write_bytes(b'fake')
        with self.assertRaises(ValueError):self.verify()
        queue=copy.deepcopy(self.queue);queue['items'][1]['output_directory']=queue['items'][0]['output_directory']
        with self.assertRaises(ValueError):self.verify(queue)

    def test_actual_queue_clock_and_branch_rejected(self):
        for field,value in [('Time Span Start','0.32'),('Time Span Duration','0.08'),('Time Span End','0.08')]:
            queue=copy.deepcopy(self.queue);queue['items'][0]['render_settings'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(queue)
        queue=copy.deepcopy(self.queue);queue['items'][0]['branch']='raw_roto'
        with self.assertRaises(ValueError):self.verify(queue)

    def test_missing_host_readback_or_input_confirmation_rejected(self):
        for field in ('review_layer_clock_readback','foreground_input_preserve_rgb_confirmed'):
            queue=copy.deepcopy(self.queue);queue[field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(queue)

    def test_changed_aep_or_input_validation_binding_rejected(self):
        self.base.write_bytes(b'CHANGED AEP')
        with self.assertRaises(ValueError):self.verify()
        self.base.write_bytes(b'FAKE AEP: not a native project')
        with self.validation_path.open('a') as stream:stream.write(' ')
        with self.assertRaises(ValueError):self.verify()

    def test_changed_original_input_rejected(self):
        Path(self.validation['frames'][8]['inputs']['foreground']['path']).write_bytes(b'CHANGED F')
        with self.assertRaises(ValueError):self.verify()

    def test_wrong_float_storage_or_dimensions_rejected(self):
        header={'width':4096,'height':2160,'compression_code':3,'channel_pixel_types':{c:2 for c in 'RGBA'}}
        for field,value in [('width',2048),('compression_code',4),('channel_pixel_types',{c:1 for c in 'RGBA'})]:
            wrong=dict(header);wrong[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(header=wrong)

    def test_tolerance_is_not_bit_exact_and_a_squared_fails(self):
        pixels=self.rgba.copy();pixels[0,1,3]=np.nextafter(self.a[0,1],np.float32(1))
        result=q.review_numeric(pixels,self.f,self.a)
        self.assertFalse(result['A_bit_exact']);self.assertLess(result['A_max_error'],q.A_TOL)
        pixels=self.rgba.copy();pixels[:,:,:3]*=self.a[:,:,None]
        with self.assertRaises(ValueError):q.review_numeric(pixels,self.f,self.a)
        pixels=self.rgba.copy();pixels[0,0,0]=np.nan
        with self.assertRaises(ValueError):q.review_numeric(pixels,self.f,self.a)

    def test_validation_clock_guard_used_for_review(self):
        self.validation['fps']=30;self.save(self.validation_path,self.validation)
        self.queue['input_validation_sha256']=q.sha(self.validation_path)
        with self.assertRaises(ValueError):self.verify()

    def make_queue(self,phase,proof=None):
        with contextlib.redirect_stdout(io.StringIO()):
            q.make_queue(SimpleNamespace(settings=str(self.settings_path),phase=phase,probe_proof=proof))
        return json.loads((self.delivery/'queues'/phase/'queue-config.json').read_text())

    def test_review_unique_five_and_probe_full_preserved(self):
        review=self.make_queue('review');probe=self.make_queue('probe')
        proof=self.root/'probe.json'
        self.save(proof,{'status':'PASS_NATIVE_ORIGINAL_F_A_PROBE','A_max_error':0,'RGB_max_error':0,
                         'base_project_sha256':q.sha(self.base),'input_validation_sha256':q.sha(self.validation_path)})
        full=self.make_queue('full',str(proof))
        self.assertEqual([s['first'] for s in review['specs']],list(q.REVIEW_FRAMES))
        self.assertEqual((review['expected_exports'],probe['expected_exports'],full['expected_exports']),(5,1,100))
        for config in (review,full):
            self.assertEqual(len(config['specs']),len({s['folder'] for s in config['specs']}))
        self.assertEqual(len(full['specs']),36)
        self.assertFalse(review['open_base_project']);self.assertTrue(full['open_base_project'])
        with self.assertRaises(FileExistsError):self.make_queue('review')

    def test_queue_requires_explicit_input_preserve_rgb(self):
        self.settings['foreground_input_preserve_rgb_confirmed']=False;self.save(self.settings_path,self.settings)
        with self.assertRaises(ValueError):self.make_queue('review')


if __name__=='__main__':unittest.main()
