"""Prepare standalone AE queues, verify native probe/review5, or collect native bytes.

No app, renderer, inference or recording is launched by this utility.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
from prepare_ae import load_settings,validate_ledger_clock
from media_precision import read_json,read_exr,sha,write_new_json
from verify_4k_ae_native_v2 import bit_exact_f32
from verify_4k_ae_native import native,check,difference,A_TOL,RGB_TOL,PATTERNS

BRANCHES=[('SOURCE_4K','source','source'),('RESTORED_FOREGROUND_RGB_4K','foreground','foreground'),
          ('RAW_LTX_PRIOR_ALPHA_4K','raw_alpha','rawprior'),('DERIVED_REFINED_ALPHA_4K','refined_alpha','refined'),
          ('ROTO_RAW_PRIOR_4K','raw_roto','raw'),('ROTO_SOURCE_REFINED_4K','source_refined','source_refined'),
          ('COMPOSITE_CLEAN_REFINED_4K','composite','composite')]
REVIEW_FRAMES=(0,8,16,32,64)

def paths(settings):
    root=Path(settings['delivery_root']);attempt=root/'attempts'/settings['attempt']
    return root,attempt,attempt/'LTX_BF16_4K_REFINED_ML_REVIEW.aep'

def make_queue(args):
    settings=load_settings(args.settings);root,attempt,base=paths(settings)
    if settings.get('foreground_input_preserve_rgb_confirmed') is not True:
        raise ValueError('Read back Interpret Footage Color > Preserve RGB on coded foreground; save the base AEP and confirm it in local settings first')
    if not base.is_file():raise FileNotFoundError(base)
    specs=[];proof={}
    if args.phase=='probe':
        specs=[{'comp':'ROTO_CLEAN_REFINED_4K','branch':'clean','folder':'clean/key-8','prefix':'clean','first':8,'count':1}]
    elif args.phase=='review':
        validation=read_json(root/'input-validation.json')
        if validation.get('full65_validated') is not True:raise ValueError('Complete producer260 input validation required')
        validate_ledger_clock(validation,validation['source_frame_range'][0])
        specs=[{'comp':'ROTO_CLEAN_REFINED_4K','branch':'clean','folder':'clean/key-'+str(key),'prefix':'clean','first':key,'count':1} for key in REVIEW_FRAMES]
    else:
        if not args.probe_proof:raise ValueError('Full queue requires an actual original-F*A native probe proof')
        proof=read_json(args.probe_proof)
        if proof.get('status')!='PASS_NATIVE_ORIGINAL_F_A_PROBE':raise ValueError('Successful numeric probe required')
        check(proof['A_max_error'],A_TOL,'Probe A');check(proof['RGB_max_error'],RGB_TOL,'Probe original F*A')
        if proof['base_project_sha256']!=sha(base):raise ValueError('Base project changed since successful probe')
        if proof['input_validation_sha256']!=sha(root/'input-validation.json'):raise ValueError('Input validation changed since probe')
        specs=[{'comp':'ROTO_CLEAN_REFINED_4K','branch':'clean','folder':'clean/full65','prefix':'clean','first':0,'count':65}]
        for comp,branch,prefix in BRANCHES:
            for key in [0,8,16,32,64]:
                specs.append({'comp':comp,'branch':branch,'folder':branch+'/key-'+str(key),'prefix':prefix,'first':key,'count':1})
    directory=root/'queues'/args.phase;directory.mkdir(parents=True,exist_ok=False)
    exports=directory/'exports'
    for spec in specs:(exports/spec['folder']).mkdir(parents=True,exist_ok=False)
    config={'phase':args.phase,'base_project':str(base),'expected_current_project':str(base if args.phase in ('probe','review') else root/'queues/probe/render.aep'),
            'open_base_project':args.phase=='full','render_project_path':str(directory/'render.aep'),
            'render_root':str(exports),'output_template':settings['output_template'],'receipt_path':str(directory/'host-queue-receipt.json'),
            'specs':specs,'expected_exports':{'probe':1,'review':5,'full':100}[args.phase],
            'probe_numeric_passed':args.phase=='full','probe_A_error':proof.get('A_max_error'),
            'probe_RGB_error':proof.get('RGB_max_error')}
    if args.phase=='review':
        config.update(base_project_sha256=sha(base),input_validation_sha256=sha(root/'input-validation.json'),
                      foreground_input_preserve_rgb_confirmed=True)
    write_new_json(directory/'queue-config.json',config)
    template=Path(__file__).with_name('ae_queue_unique.jsx').read_text(encoding='utf-8')
    with (directory/'run-in-AE.jsx').open('x',encoding='utf-8') as stream:
        stream.write(template.replace('__CONFIG_JSON__',json.dumps(config,ensure_ascii=True)))
    print(json.dumps({'prepared':str(directory/'run-in-AE.jsx'),'expected_exports':config['expected_exports'],
                      'run_after_AE_save':['aerender','-project',config['render_project_path'],'-mfr','OFF','100'],'AE_executed':False}))

def verify_probe(args):
    settings=load_settings(args.settings);root,attempt,base=paths(settings)
    validation=root/'input-validation.json';row=read_json(validation)['frames'][args.local]['inputs']
    for role in ('foreground','alpha'):
        if sha(row[role]['path'])!=row[role]['sha256']:raise ValueError('Validated producer input changed')
    f,_=read_exr(row['foreground']['path']);m,_=read_exr(row['alpha']['path'],alpha=True);a=m[:,:,3]
    pixels,evidence=native(Path(args.native_exr));expected=f*a[:,:,None]
    ae=check(difference(pixels[:,:,3],a),A_TOL,'Original A');re=check(difference(pixels[:,:,:3],expected),RGB_TOL,'Original input F*A')
    report={'status':'PASS_NATIVE_ORIGINAL_F_A_PROBE','local_index':args.local,'A_max_error':ae,'RGB_max_error':re,
            'A_bit_exact':bit_exact_f32(pixels[:,:,3],a),'RGB_bit_exact':bit_exact_f32(pixels[:,:,:3],expected),
            'base_project_sha256':sha(base),'input_validation_sha256':sha(validation),'native_file':evidence,
            'color_contract':'Coded BT709 data. Native associated RGB must equal original F*A; no gamma workaround or alpha alteration.'}
    write_new_json(args.receipt,report);print(json.dumps({k:report[k] for k in ('status','A_max_error','RGB_max_error','A_bit_exact','RGB_bit_exact')}))

def review_numeric(pixels,foreground,alpha):
    if (pixels.dtype!=np.float32 or foreground.dtype!=np.float32 or alpha.dtype!=np.float32
        or pixels.shape!=alpha.shape+(4,) or foreground.shape!=alpha.shape+(3,)
        or not all(np.isfinite(a).all() for a in (pixels,foreground,alpha))):
        raise ValueError('Finite matching native F32 review arrays required')
    expected=foreground*alpha[:,:,None]
    ae=check(difference(pixels[:,:,3],alpha),A_TOL,'Review original A')
    re=check(difference(pixels[:,:,:3],expected),RGB_TOL,'Review original F*A')
    return {'A_max_error':ae,'RGB_max_error':re,'A_bit_exact':bit_exact_f32(pixels[:,:,3],alpha),
            'RGB_bit_exact':bit_exact_f32(pixels[:,:,:3],expected)}

def verify_review(args):
    import OpenImageIO  # Require the validated native decoder, not a silent fallback.
    settings=load_settings(args.settings);root,attempt,base=paths(settings)
    validation_path=root/'input-validation.json';validation=read_json(validation_path);queue=read_json(args.queue_receipt)
    if queue.get('ok') is not True or queue.get('phase')!='review':raise ValueError('Actual review-queue host receipt required')
    if queue.get('review_layer_clock_readback') is not True:raise ValueError('Actual single-ALPHA/layer clock readback required')
    if queue.get('foreground_input_preserve_rgb_confirmed') is not True:raise ValueError('Foreground input PreserveRGB confirmation required')
    if queue.get('base_project_sha256')!=sha(base):raise ValueError('Base AEP changed since review queue preparation')
    if queue.get('input_validation_sha256')!=sha(validation_path):raise ValueError('Input-validation binding changed')
    if validation.get('full65_validated') is not True:raise ValueError('Complete producer260 input validation required')
    rows=validate_ledger_clock(validation,validation['source_frame_range'][0])
    items=queue.get('items',[])
    if len(items)!=5:raise ValueError('Exactly five review items required')
    keys=[item.get('first_local_frame') for item in items]
    if any(type(k) is not int for k in keys) or set(keys)!=set(REVIEW_FRAMES) or len(set(keys))!=5:
        raise ValueError('Exact unique review keys0/8/16/32/64 required')
    directories=[Path(item['output_directory']).resolve() for item in items]
    if len(set(directories))!=5:raise ValueError('Five unique review output directories required')
    results=[]
    for item,directory in zip(items,directories):
        key=item['first_local_frame'];rs=item['render_settings']
        start=float(rs['Time Span Start']);duration=float(rs['Time Span Duration'])
        if (item.get('branch')!='clean' or item.get('comp')!='ROTO_CLEAN_REFINED_4K' or item.get('frame_count')!=1
            or not np.isfinite(start) or not np.isfinite(duration)
            or abs(start-key/25)>1e-9 or abs(duration-1/25)>1e-9):
            raise ValueError('Actual review queue clock/branch mismatch')
        if 'Time Span End' in rs:
            end=float(rs['Time Span End'])
            if not np.isfinite(end) or abs(end-start-duration)>1e-9:raise ValueError('Actual review end time mismatch')
        files=list(directory.glob('*.exr'))
        if len(files)!=1:raise ValueError('Exactly one native EXR per review item required')
        inputs=rows[key]['inputs']
        for role in ('foreground','alpha'):
            if sha(inputs[role]['path'])!=inputs[role]['sha256']:raise ValueError('Review source input changed')
        f,_=read_exr(inputs['foreground']['path']);m,_=read_exr(inputs['alpha']['path'],alpha=True)
        pixels,evidence=native(files[0]);header=evidence['header']
        if (header.get('width'),header.get('height'),header.get('compression_code'))!=(4096,2160,3) or header.get('channel_pixel_types')!={c:2 for c in 'RGBA'}:
            raise ValueError('Actual native4K ZIP/FLOAT32 RGBA header required')
        results.append({'local_index':key,'source_frame_index':rows[key]['source_frame_index'],
                        'metrics':review_numeric(pixels,f,m[:,:,3]),'native_file':evidence})
    if sha(base)!=queue['base_project_sha256'] or sha(validation_path)!=queue['input_validation_sha256']:
        raise ValueError('Review binding changed during verification')
    report={'status':'PASS_NATIVE_REVIEW5_SAMPLES_ONLY','frames_checked':5,'local_frames':list(REVIEW_FRAMES),
            'all65_native_verified':False,'source_RGB8_native_verified':False,'opaque_F_native_verified':False,
            'base_project_sha256':sha(base),'input_validation_sha256':sha(validation_path),
            'queue_receipt_sha256':sha(args.queue_receipt),'alpha_tolerance':A_TOL,'RGB_tolerance':RGB_TOL,
            'A_bit_exact_all5':all(r['metrics']['A_bit_exact'] for r in results),
            'RGB_bit_exact_all5':all(r['metrics']['RGB_bit_exact'] for r in results),
            'A_max_error':max(r['metrics']['A_max_error'] for r in results),
            'RGB_max_error':max(r['metrics']['RGB_max_error'] for r in results),
            'visual_approval':False,'required_visual_review':'All65 producer/motion frames, separately from this five-sample native check.',
            'scope':'Repeat-shot AEP review only. Not an all65 native precision certificate; new model/AE version/color settings require full100.',
            'frames':results}
    write_new_json(args.receipt,report)
    print(json.dumps({k:report[k] for k in ('status','frames_checked','all65_native_verified','A_max_error','RGB_max_error')}))

def collect(args):
    validation=read_json(args.validation);queue=read_json(args.queue_receipt)
    if queue.get('ok') is not True or queue.get('phase')!='full':raise ValueError('Actual full-queue host receipt required')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    lookup={}
    for row in validation['frames']:
        lookup.setdefault(row['inputs']['alpha']['alpha_pixel_sha256'],[]).append(row['local_index'])
    records=[];used=set()
    for item in queue['items']:
        first=round(float(item['render_settings']['Time Span Start'])*25)
        count=round(float(item['render_settings']['Time Span Duration'])*25)
        if first!=item['first_local_frame'] or count!=item['frame_count']:raise ValueError('Actual queue clock mismatch')
        files=sorted(Path(item['output_directory']).glob('*.exr'))
        if len(files)!=count:raise ValueError('Queue-item output count mismatch')
        for source in files:
            branch=item['branch'];local=first
            if branch=='clean':
                pixels,_=native(source);key=hashlib.sha256(np.ascontiguousarray(pixels[:,:,3]).tobytes()).hexdigest()
                candidates=lookup.get(key,[])
                if len(candidates)!=1:raise ValueError('Clean alpha hash does not unambiguously identify one original frame')
                local=candidates[0]
                if not first<=local<first+count:raise ValueError('Native clean frame outside actual time span')
            elif count!=1:raise ValueError('Auxiliary item must be one source frame')
            if (branch,local) in used:raise ValueError('Duplicate canonical frame')
            used.add((branch,local));target=output/(PATTERNS[branch]%local);target.parent.mkdir(parents=True,exist_ok=True)
            before=sha(source);shutil.copy2(source,target)
            if sha(target)!=before:raise ValueError('Native byte copy changed')
            records.append({'branch':branch,'local_index':local,'origin':str(source),'canonical':str(target),'sha256':before})
    expected={(b,i) for b in PATTERNS for i in (range(65) if b=='clean' else [0,8,16,32,64])}
    if used!=expected:raise ValueError('Canonical100 matrix incomplete')
    write_new_json(output/'origin-ledger.json',{'status':'COLLECTED100_NATIVE_BYTES_PIXEL_VERIFICATION_PENDING','files':records})
    print(json.dumps({'native_files':len(records),'pixel_processing':'None; byte-exact copies only','next':'Run verify_4k_ae_native_v2.py'}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--settings',default=str(Path(__file__).with_name('settings.example.json')))
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('make-queue');q.add_argument('--phase',choices=['probe','review','full'],required=True);q.add_argument('--probe-proof')
    v=sub.add_parser('verify-probe');v.add_argument('--native-exr',required=True);v.add_argument('--local',type=int,default=8);v.add_argument('--receipt',required=True)
    c=sub.add_parser('collect');c.add_argument('--validation',required=True);c.add_argument('--queue-receipt',required=True);c.add_argument('--output',required=True)
    r=sub.add_parser('verify-review');r.add_argument('--queue-receipt',required=True);r.add_argument('--receipt',required=True)
    args=p.parse_args();{'make-queue':make_queue,'verify-probe':verify_probe,'verify-review':verify_review,'collect':collect}[args.command](args)
if __name__=='__main__':main()
