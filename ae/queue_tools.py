"""Prepare standalone AE queue JSX, check one native probe, or collect native bytes.

No app, renderer, inference or recording is launched by this utility.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
from prepare_ae import load_settings
from media_precision import read_json,read_exr,sha,write_new_json
from verify_4k_ae_native_v2 import bit_exact_f32
from verify_4k_ae_native import native,check,difference,A_TOL,RGB_TOL,PATTERNS

BRANCHES=[('SOURCE_4K','source','source'),('RESTORED_FOREGROUND_RGB_4K','foreground','foreground'),
          ('RAW_LTX_PRIOR_ALPHA_4K','raw_alpha','rawprior'),('DERIVED_REFINED_ALPHA_4K','refined_alpha','refined'),
          ('ROTO_RAW_PRIOR_4K','raw_roto','raw'),('ROTO_SOURCE_REFINED_4K','source_refined','source_refined'),
          ('COMPOSITE_CLEAN_REFINED_4K','composite','composite')]

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
    config={'phase':args.phase,'base_project':str(base),'expected_current_project':str(base if args.phase=='probe' else root/'queues/probe/render.aep'),
            'open_base_project':args.phase=='full','render_project_path':str(directory/'render.aep'),
            'render_root':str(exports),'output_template':settings['output_template'],'receipt_path':str(directory/'host-queue-receipt.json'),
            'specs':specs,'expected_exports':1 if args.phase=='probe' else 100,
            'probe_numeric_passed':args.phase=='full','probe_A_error':proof.get('A_max_error'),
            'probe_RGB_error':proof.get('RGB_max_error')}
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
    q=sub.add_parser('make-queue');q.add_argument('--phase',choices=['probe','full'],required=True);q.add_argument('--probe-proof')
    v=sub.add_parser('verify-probe');v.add_argument('--native-exr',required=True);v.add_argument('--local',type=int,default=8);v.add_argument('--receipt',required=True)
    c=sub.add_parser('collect');c.add_argument('--validation',required=True);c.add_argument('--queue-receipt',required=True);c.add_argument('--output',required=True)
    args=p.parse_args();{'make-queue':make_queue,'verify-probe':verify_probe,'collect':collect}[args.command](args)
if __name__=='__main__':main()
