"""The tested native4K28-tile F32/r48 method, split into GPU and CPU workers."""
from pathlib import Path
import gc, json, os, shutil, sys, time
from .common import REPO, command, read_json, save_json, sha
from .media import bit_exact, make_trimap, pixel_sha, read_exr, require_f32, tile_geometry, white_alpha, write_exr

METHOD = 'LTX_BF16_FHD_PRIOR__VITMATTE_BASE_F32_4K_R48__ML_CODED_V1'
ML = dict(regularization=1e-5, n_small_iterations=10, n_big_iterations=2,
          small_size=32, return_background=False, gradient_weight=1.)

def verify_inputs(row):
    for item in row['inputs'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input SHA changed')

def tiled_frame(out, row, cfg):
    import collections
    import numpy as np
    import torch
    from PIL import Image
    from transformers import VitMatteConfig, VitMatteImageProcessor, VitMatteForImageMatting
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    model_dir = Path(cfg['vitmatte_model_dir'])
    provenance = read_json(REPO / 'config/model-provenance.json')['vitmatte']
    for item in provenance['files']:
        if sha(model_dir / item['name']) != item['sha256']:
            raise ValueError('Pinned ViTMatte model/config SHA mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this validated profile; no fallback')
    free,total=torch.cuda.mem_get_info()
    if free < 8*1024**3:
        raise RuntimeError('F32 tile worker needs8GiB free VRAM; no fallback')
    torch.cuda.set_per_process_memory_fraction(min(.8,(free-2*1024**3)/total))
    torch.manual_seed(42); torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    model=VitMatteForImageMatting(VitMatteConfig.from_pretrained(model_dir,local_files_only=True)).float().eval()
    state=torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Expected a tensor-only state dictionary')
    if any(v.dtype!=torch.float32 for v in state.values() if v.is_floating_point()):
        raise ValueError('Native F32 checkpoint required')
    weight_dtypes=dict(collections.Counter(str(v.dtype) for v in state.values()))
    model.load_state_dict(state,strict=True);del state;gc.collect();model.to('cuda',dtype=torch.float32)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Model parameter precision changed')
    processor=VitMatteImageProcessor.from_pretrained(model_dir,local_files_only=True)
    rgb=np.asarray(Image.open(row['inputs']['source']['path']).convert('RGB'))
    prior=read_exr(row['inputs']['raw_prior']['path'])['A']
    if rgb.shape!=(2160,4096,3) or prior.shape!=(2160,4096):raise ValueError('Native4K grid required')
    trimap=make_trimap(prior);Image.fromarray(trimap).save(out/'trimap.png')
    xs,ys,weight,norm=tile_geometry();accum=np.zeros_like(norm);coverage=np.zeros(norm.shape,np.uint8);metrics=[]
    for y in ys:
        for x in xs:
            image=Image.fromarray(np.ascontiguousarray(rgb[y:y+1024,x:x+1024]))
            seeds=Image.fromarray(np.ascontiguousarray(trimap[y:y+1024,x:x+1024]))
            data=processor(images=image,trimaps=seeds,return_tensors='pt')['pixel_values']
            if tuple(data.shape)!=(1,4,1024,1024) or data.dtype!=torch.float32:raise ValueError('Processor grid/dtype changed')
            data=data.to('cuda',dtype=torch.float32);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):prediction=model(pixel_values=data).alphas
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            if prediction.dtype!=torch.float32:raise ValueError('Prediction is not FLOAT32')
            alpha=prediction[0,0].cpu().numpy().copy();require_f32(alpha)
            accum[y:y+1024,x:x+1024]+=alpha*weight;coverage[y:y+1024,x:x+1024]+=1
            metrics.append({'xy':[x,y],'seconds':seconds,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
            print(json.dumps({'source_frame':row['source_frame_index'],'tile':len(metrics),'total':28}),flush=True)
            del data,prediction,alpha;gc.collect()
    alpha=accum/norm;require_f32(alpha)
    if (coverage==0).any() or alpha.min()<0 or alpha.max()>1.000001:raise ValueError('Coverage/range failed; do not clamp')
    np.save(out/'alpha.npy',alpha,allow_pickle=False)
    save_json(out/'inference.json',{'method':METHOD,'A_pixel_sha256':pixel_sha(alpha),'checkpoint_dtypes':weight_dtypes,
              'prediction_dtype':'float32','autocast':False,'TF32':False,'tiles':metrics,'normalized_overlap':True,
              'coverage_min':int(coverage.min()),'coverage_max':int(coverage.max()),'weight_pixel_sha256':pixel_sha(weight),
              'weight_sum_pixel_sha256':pixel_sha(norm),'final_alpha_threshold_erosion_feather':False})

def foreground_frame(out,row):
    import numpy as np
    import psutil
    from PIL import Image
    from pymatting import estimate_foreground_ml
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('ML worker needs4GiB free RAM')
    alpha=np.load(out/'alpha.npy',allow_pickle=False);require_f32(alpha);alpha.setflags(write=False);before=pixel_sha(alpha)
    rgb=np.asarray(Image.open(row['inputs']['source']['path']).convert('RGB')).astype(np.float64)/255.
    foreground=estimate_foreground_ml(rgb,alpha.astype(np.float64),**ML).astype(np.float32)
    require_f32(foreground)
    if pixel_sha(alpha)!=before:raise ValueError('Foreground estimation modified alpha')
    np.save(out/'foreground.npy',foreground,allow_pickle=False)
    save_json(out/'foreground.json',{'method':METHOD,'A_unchanged':True,'A_pixel_sha256':before,'F_pixel_sha256':pixel_sha(foreground),
              'parameters':ML,'RGB_space':'BT709 full RGB code values, no extra EOTF','foreground_working_dtype':'float64','output_dtype':'float32'})

def pack_frame(root,out,row):
    import numpy as np
    local=row['local_index'];index=local+1
    alpha=np.load(out/'alpha.npy',allow_pickle=False);foreground=np.load(out/'foreground.npy',allow_pickle=False)
    inference=read_json(out/'inference.json');fg_receipt=read_json(out/'foreground.json')
    if pixel_sha(alpha)!=inference['A_pixel_sha256'] or pixel_sha(alpha)!=fg_receipt['A_pixel_sha256'] or pixel_sha(foreground)!=fg_receipt['F_pixel_sha256']:
        raise ValueError('F32 working data changed')
    paths={'source':root/f'assets/source/source_{index:05d}.png',
           'raw_prior':root/f'assets/raw_prior_rgba_float32/alpha_{index:05d}.exr',
           'alpha':root/f'assets/alpha_rgba_float32/alpha_{index:05d}.exr',
           'foreground':root/f'assets/foreground_rgb_float32/foreground_{index:05d}.exr'}
    for key in ('source','raw_prior'):
        paths[key].parent.mkdir(parents=True,exist_ok=True)
        if paths[key].exists():
            if sha(paths[key])!=row['inputs'][key]['sha256']:raise ValueError('Existing copy changed')
        else:shutil.copyfile(row['inputs'][key]['path'],paths[key])
    white_alpha(paths['alpha'],alpha)
    write_exr(paths['foreground'],{c:foreground[:,:,j] for j,c in enumerate('RGB')},'BT709-code-values')
    result={k:row[k] for k in ('local_index','source_frame_index','file_index','time_seconds')}
    result.update(method=METHOD,inputs={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
                  A_pixel_sha256=pixel_sha(alpha),F_pixel_sha256=pixel_sha(foreground),F32_EXR_readback_bit_exact=True,
                  foreground_has_alpha=False,alpha_RGB_white=True,raw_prior_preserved=True)
    save_json(root/f'receipts/frame_{index:05d}.json',result)
    return result

def worker(phase,root,local):
    root=Path(root).resolve();plan=read_json(root/'plan.json');row=next(r for r in plan['frames'] if r['local_index']==local)
    out=root/f'working/local{local:02d}';out.mkdir(parents=True,exist_ok=True);verify_inputs(row)
    if phase=='tile':tiled_frame(out,row,plan['config'])
    elif phase=='foreground':foreground_frame(out,row)
    else:return pack_frame(root,out,row)
    verify_inputs(row)
    return {'phase':phase,'local_index':local,'status':'SUCCEEDED'}

def run_refine(run,root):
    run=Path(run).resolve();root=Path(root).resolve();prior=read_json(run/'prior-ledger.json');cfg=read_json(run/'config.json')
    if (prior['width'],prior['height'],prior['frame_count'],prior['fps'])!=(4096,2160,65,25):raise ValueError('Validated65/native4K/25fps ledger required')
    plan={'method':METHOD,'config':cfg,'frames':prior['frames'],'prior_ledger_sha256':sha(run/'prior-ledger.json')}
    if root.exists():
        if read_json(root/'plan.json')!=plan:raise ValueError('Existing run differs; preserve and use a new output directory')
    else:
        root.mkdir(parents=True);save_json(root/'plan.json',plan)
    rows=[]
    for row in prior['frames']:
        verify_inputs(row);local=row['local_index'];receipt=root/f'receipts/frame_{local+1:05d}.json'
        if receipt.exists():
            completed=read_json(receipt)
            if completed['method']!=METHOD or completed['source_frame_index']!=row['source_frame_index']:raise ValueError('Prior frame receipt mismatch')
            verify_inputs(completed);rows.append(completed);continue
        for phase,exe in [('tile',cfg['gpu_python']),('foreground',cfg['cpu_python']),('pack',cfg['gpu_python'])]:
            log=root/f'working/local{local:02d}/{phase}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('ab') as stream:
                command([exe,'-B','-m','pipeline','_worker',phase,'--out',root,'--local',str(local)],cwd=REPO,
                        stdout=stream,stderr=stream,timeout=1200)
        rows.append(read_json(receipt));print(json.dumps({'completed_frame':local+1,'total':65}),flush=True)
    ledger={'method':METHOD,'width':4096,'height':2160,'fps':25,'frame_count':65,'frames':rows,
            'RGB':'original full-range BT709 code values; ML straight RGB3/noA; no extra EOTF',
            'alpha':'derivedF32 linear coverage, whiteRGBA; raw mapped prior preserved separately',
            'frame_clock':'local0..64 = source75..139 by default; source index is ledger-authoritative',
            'visual_approval':False}
    save_json(root/'frame-ledger.json',ledger)
    save_json(root/'completion.json',{'status':'ALL65_F32_TYPED_EXPORTS_COMPLETE_VISUAL_PENDING','files':260,'ledger_sha256':sha(root/'frame-ledger.json')})
    return {'status':'COMPLETE_REQUIRES_TEMPORAL_VISUAL_QA','ledger':str(root/'frame-ledger.json')}
