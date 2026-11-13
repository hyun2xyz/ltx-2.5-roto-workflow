"""CPU proof for65 native-clean frames plus5 samples of seven other branches.

100 selected native4K EXRs total. Shared numerical helpers are retained from the original v1.
No application, render, recording or inference is launched here.
"""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
from media_precision import read_exr, sha, write_new_json
from verify_4k_ae_native import A_TOL, RGB_TOL, COMPOSITE_TOL, PATTERNS, read_json, check, difference, native

KEY_FRAMES = (0,8,16,32,64)

def frame_indices(branch):return tuple(range(65)) if branch=='clean' else KEY_FRAMES

def bit_exact_f32(actual,expected):
    a=np.ascontiguousarray(actual,dtype=np.float32).view(np.uint32)
    b=np.ascontiguousarray(expected,dtype=np.float32).view(np.uint32)
    return bool(np.array_equal(a,b))

def verify(args):
    import OpenImageIO  # Require the independently verified native truth decoder.
    inputs_report=read_json(args.validation);build=read_json(args.build_receipt)
    if inputs_report.get('status')!='PASS_ALL65_INPUT_FILES_F32_CHANNEL_CONTRACT' or inputs_report.get('full65_validated') is not True:
        raise ValueError('Full65 producer260-file input validation required')
    if build.get('ok') is not True or build.get('input_validation_sha256')!=sha(args.validation):
        raise ValueError('Actual host builder receipt must bind this complete input validation')
    project=build['project']
    if (project['bits_per_channel']!=32 or project['working_space_raw'] not in ('','None')
        or project['linear_blending'] is not False or project['linearize_working_space'] is not False):
        raise ValueError('Native32bpc unmanaged project readback required')
    expected={'ROTO_RAW_PRIOR_4K','ROTO_SOURCE_REFINED_4K','ROTO_CLEAN_REFINED_4K'}
    tracks=build['track_mattes']
    if len(tracks)!=3 or {r['comp'] for r in tracks}!=expected:raise ValueError('Three independent ALPHA branches required')
    for row in tracks:
        if row['alpha_applications_in_branch']!=1 or row['rgb_input_has_alpha'] is not False or row.get('real_alpha_readback') is not True:
            raise ValueError('Single real ALPHA application not proved')
    composites=build.get('composite_branches',[])
    if len(composites)!=15 or any(r['second_track_matte'] is not False for r in composites):
        raise ValueError('Direct associated-cutout comps must have no second matte')
    if len(inputs_report['frames'])!=65:raise ValueError('Exactly65 producer frames required')
    root=Path(args.render_root).resolve(strict=True)
    source_start=inputs_report['source_frame_range'][0]
    plan={key:frame_indices(key) for key in PATTERNS}
    for key,pattern in PATTERNS.items():
        directory=root/Path(pattern).parent
        expected_names={Path(pattern%i).name for i in plan[key]}
        if {p.name for p in directory.glob('*.exr')}!=expected_names:
            raise ValueError('Exact native export set mismatch: '+key)
    result_frames=[];total_soft=0;max_A2_difference=0.;all_A_bit_exact=True
    max_A_error=0.;max_clean_RGB_error=0.;native_files_checked=0
    for local,row in enumerate(inputs_report['frames']):
        if row['local_index']!=local or row['source_frame_index']!=source_start+local:raise ValueError('Source/local index mismatch')
        inputs=row['inputs']
        for item in inputs.values():
            if sha(item['path'])!=item['sha256']:raise ValueError('A validated producer file changed')
        foreground,_=read_exr(inputs['foreground']['path'],alpha=False)
        decoded,_=read_exr(inputs['alpha']['path'],alpha=True);alpha=decoded[:,:,3].copy();del decoded
        clean,evidence_clean=native(root/(PATTERNS['clean']%local));native_files_checked+=1
        a_error=check(difference(clean[:,:,3],alpha),A_TOL,'Clean native A vs original refined A')
        a_exact=bit_exact_f32(clean[:,:,3],alpha)
        expected_RGB=foreground*alpha[:,:,None]
        rgb_error=check(difference(clean[:,:,:3],expected_RGB),RGB_TOL,'Clean associatedRGB vs original input F*A')
        rgb_exact=bit_exact_f32(clean[:,:,:3],expected_RGB);del expected_RGB
        soft=(alpha>.05)&(alpha<.95);soft_count=int(soft.sum());total_soft+=soft_count
        a2_error=float(np.max(np.abs(clean[:,:,3][soft].astype(np.float64)-(alpha[soft]*alpha[soft]).astype(np.float64)))) if soft_count else 0.
        max_A2_difference=max(max_A2_difference,a2_error);all_A_bit_exact=all_A_bit_exact and a_exact
        max_A_error=max(max_A_error,a_error);max_clean_RGB_error=max(max_clean_RGB_error,rgb_error)
        metrics={'clean_A_max_error':a_error,'clean_A_bit_exact':a_exact,
                 'clean_A_tolerance_pass':True,'clean_RGB_vs_original_F_times_original_A_max_error':rgb_error,
                 'clean_RGB_bit_exact_to_F32_product':rgb_exact,'clean_RGB_tolerance_pass':True,
                 'soft_coverage_pixels':soft_count,'native_A_vs_A_squared_max_difference':a2_error}
        evidence={'clean':evidence_clean}
        if local in KEY_FRAMES:
            with Image.open(inputs['source']['path']) as image:source8=np.asarray(image).copy()
            src,evidence['source']=native(root/(PATTERNS['source']%local))
            recovered=np.rint(np.clip(src[:,:,:3],0,1).astype(np.float64)*255).astype(np.uint8)
            if not np.array_equal(recovered,source8):raise ValueError('Sampled originalRGB8 native information loss')
            metrics['sample_source_RGB8_byte_exact']=True
            metrics['sample_source_normalized_float_max_error']=difference(src[:,:,:3],source8.astype(np.float32)/np.float32(255))
            metrics['sample_source_A_one_max_error']=check(difference(src[:,:,3],np.ones_like(alpha)),A_TOL,'Sample source A')
            source_native=src[:,:,:3].copy();del src,source8,recovered
            fg,evidence['foreground']=native(root/(PATTERNS['foreground']%local))
            metrics['sample_opaque_F_import_max_error']=check(difference(fg[:,:,:3],foreground),RGB_TOL,'Sample opaqueF import')
            metrics['sample_opaque_F_import_bit_exact']=bit_exact_f32(fg[:,:,:3],foreground)
            metrics['sample_opaque_F_A_one_max_error']=check(difference(fg[:,:,3],np.ones_like(alpha)),A_TOL,'Sample foreground A');del fg
            prior_rgba,_=read_exr(inputs['raw_prior']['path'],alpha=True);prior=prior_rgba[:,:,3].copy();del prior_rgba
            for kind,a in [('raw_alpha',prior),('refined_alpha',alpha)]:
                rendered,evidence[kind]=native(root/(PATTERNS[kind]%local))
                metrics['sample_'+kind+'_max_error']=check(difference(rendered[:,:,3],a),A_TOL,kind)
                metrics['sample_'+kind+'_bit_exact']=bit_exact_f32(rendered[:,:,3],a);del rendered
            for kind,a in [('raw_roto',prior),('source_refined',alpha)]:
                rendered,evidence[kind]=native(root/(PATTERNS[kind]%local))
                metrics['sample_'+kind+'_A_max_error']=check(difference(rendered[:,:,3],a),A_TOL,kind+' A')
                metrics['sample_'+kind+'_associated_RGB_max_error']=check(difference(rendered[:,:,:3],source_native*a[:,:,None]),RGB_TOL,kind+' RGB association')
                del rendered
            composite,evidence['composite']=native(root/(PATTERNS['composite']%local))
            expected_comp=clean[:,:,:3].astype(np.float64)+np.float64(.32)*(1-clean[:,:,3].astype(np.float64))[:,:,None]
            metrics['sample_gray_composite_max_error']=check(difference(composite[:,:,:3],expected_comp),COMPOSITE_TOL,'Sample gray composite')
            metrics['sample_composite_A_one_max_error']=check(difference(composite[:,:,3],np.ones_like(alpha)),A_TOL,'Sample composite A')
            native_files_checked+=7
            del composite,expected_comp,prior,source_native
        result_frames.append({'local_index':local,'source_frame_index':source_start+local,'sampled_auxiliary_branches':local in KEY_FRAMES,
                              'metrics':metrics,'native_files':evidence})
        del clean,foreground,alpha,soft
    if native_files_checked!=100:raise ValueError('Expected100 native4K exports total')
    if total_soft==0 or max_A2_difference<=1e-3:raise ValueError('No meaningful A-vs-A-squared evidence')
    result={'status':'PASS_NATIVE_CLEAN65_TOLERANCE_AND_KEY5_BRANCH_PROOFS','visual_approval':False,
            'input_validation_sha256':sha(args.validation),'build_receipt_sha256':sha(args.build_receipt),
            'producer_files_SHA_bound':260,'native_files_checked':100,'native_clean_frames_checked':65,
            'sampled_auxiliary_local_frames':list(KEY_FRAMES),'sampled_auxiliary_source_frames':[source_start+i for i in KEY_FRAMES],
            'source_RGB8_native_byte_exact_sample_count':5,'source_RGB8_native_byte_exact_all_5_samples':True,
            'opaque_F32_import_native_sample_count':5,'unsampled_source_F_raw_auxiliary_native_frames':60,
            'native_clean_A_tolerance_pass_all65':True,'native_clean_A_bit_exact_all65':all_A_bit_exact,
            'native_clean_A_max_error_all65':max_A_error,'native_clean_RGB_vs_original_input_F_times_A_max_error_all65':max_clean_RGB_error,
            'soft_pixels_A_squared_regression_checked':total_soft,'maximum_native_A_vs_A_squared_difference':max_A2_difference,
            'native_real_ALPHA_once_per_branch':True,'tolerances':{'alpha':A_TOL,'RGB':RGB_TOL,'composite':COMPOSITE_TOL},
            'exactness_statement':'bit_exact compares literal F32 uint32 patterns; tolerance PASS alone is never labelled bit-exact.',
            'association':'All65 native-clean RGB are compared directly with original producer F32 F*original refined A once. Auxiliary sampled association checks and gray composite use actual native values to isolate native math.',
            'visual_scope':'Four-background visual inspection uses selected samples; whole-shot temporal review derives from the complete65-frame native-clean output. No visual approval is supplied by this technical check.',
            'decoder':'OpenImageIO by channel names; all100 actual native headers must be ZIP3/FLOAT2 RGBA',
            'frames':result_frames}
    write_new_json(args.receipt,result)
    print(json.dumps({k:result[k] for k in ['status','native_files_checked','native_clean_frames_checked','native_clean_A_bit_exact_all65','native_clean_A_max_error_all65','source_RGB8_native_byte_exact_sample_count']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['validation','build-receipt','render-root','receipt']:parser.add_argument('--'+name,required=True)
    verify(parser.parse_args())
