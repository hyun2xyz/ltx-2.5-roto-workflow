"""Portable65-frame4K AE profile; prepares code/validates inputs, never starts AE."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
from PIL import Image
from media_precision import exr_header,png_header,sha,write_new_json
PACKAGE=Path.cwd()/"outputs/ae"
FOLDERS={"source":("source","source",".png"),"alpha":("alpha_rgba_float32","alpha",".exr"),"foreground":("foreground_rgb_float32","foreground",".exr"),"raw_prior":("raw_prior_rgba_float32","alpha",".exr")}
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def in_package(path):
    path=Path(path).resolve()
    if not path.is_relative_to(PACKAGE.resolve()) or path==PACKAGE.resolve():
        raise ValueError('Output must be below the owned new4K delivery package')
    return path

def read_float_exr(path, channels):
    import OpenImageIO as oiio
    header=exr_header(path)
    if (header['width'],header['height'])!=(4096,2160):raise ValueError('4K dimensions required')
    if set(header['channel_pixel_types'])!=set(channels) or not header['all_channels_float32']:
        raise ValueError('Exact named FLOAT32 channel set required: '+str(path))
    if header['compression_code']!=3:raise ValueError('ZIP16 compression required')
    inp=oiio.ImageInput.open(str(path))
    if inp is None:raise ValueError('OIIO cannot open EXR')
    try:
        spec=inp.spec();names=list(spec.channelnames)
        pixels=np.asarray(inp.read_image(format=oiio.FLOAT),dtype=np.float32)
    finally:inp.close()
    order=[names.index(c) for c in channels]
    pixels=pixels[:,:,order].copy()
    if not np.isfinite(pixels).all():raise ValueError('Nonfinite F32 pixels')
    return pixels,header

def inspect_asset(path, role, expected_sha=None):
    path=Path(path)
    actual_sha=sha(path)
    if expected_sha is not None and actual_sha!=expected_sha:raise ValueError('Producer SHA mismatch: '+str(path))
    if role=='source':
        header=png_header(path)
        if (header['width'],header['height'],header['bit_depth'],header['color_type'])!=(4096,2160,8,2):
            raise ValueError('Exact original-grid RGB8 PNG without A required')
        with Image.open(path) as im:pixels=np.asarray(im).copy()
        if pixels.dtype!=np.uint8 or pixels.shape!=(2160,4096,3):raise ValueError('Source decoder changed grid')
    else:
        names='RGB' if role=='foreground' else 'RGBA'
        pixels,header=read_float_exr(path,names)
        if role!='foreground':
            if not np.all(pixels[:,:,:3]==np.float32(1)):raise ValueError('Matte RGB must be exactly white')
            if pixels[:,:,3].min()<0 or pixels[:,:,3].max()>1:raise ValueError('Coverage A outside0..1')
    result={'path':str(path.resolve()),'sha256':actual_sha,'bytes':path.stat().st_size,
            'header':header,'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
            'shape':list(pixels.shape),'dtype':str(pixels.dtype),'minimum':float(pixels.min()),'maximum':float(pixels.max())}
    if role in ('alpha','raw_prior'):
        result['alpha_pixel_sha256']=hashlib.sha256(np.ascontiguousarray(pixels[:,:,3]).tobytes()).hexdigest()
        result['white_rgb_exact']=True
    if role=='foreground':result['alpha_channel_absent']=True
    return result

def validate_ledger_clock(ledger,source_start):
    actual=tuple(ledger.get(k) for k in ('width','height','fps','frame_count'))
    if actual!=(4096,2160,25,65):
        raise ValueError('This AE profile requires ledger width4096/height2160/fps25/frame_count65')
    rows=ledger.get('frames',[])
    if len(rows)!=65:raise ValueError('Producer ledger must contain exactly65 frames')
    for local,row in enumerate(rows):
        if row['local_index']!=local or row['source_frame_index']!=source_start+local:
            raise ValueError('Source/local frame mapping mismatch')
        if 'time_seconds' in row and (not np.isfinite(float(row['time_seconds'])) or abs(float(row['time_seconds'])-local/25)>1e-9):
            raise ValueError('Row time_seconds must equal local_index/25')
    return rows

def verify(args):
    root=Path(args.producer_root).resolve(strict=True);ledger_path=root/'frame-ledger.json'
    ledger=read(ledger_path);source_start=args.source_start;rows=validate_ledger_clock(ledger,source_start)
    for role,(folder,prefix,ext) in FOLDERS.items():
        expected={f'{prefix}_{i:05d}{ext}' for i in range(1,66)}
        actual={p.name for p in (root/'assets'/folder).glob(prefix+'_*'+ext)}
        if actual!=expected:raise ValueError('Exact contiguous00001..00065 file set required: '+role)
    outputs=[]
    for local,row in enumerate(rows):
        if row['local_index']!=local or row['source_frame_index']!=source_start+local:raise ValueError('Source/local frame mapping mismatch')
        checked={}
        for role,(folder,prefix,ext) in FOLDERS.items():
            path=root/'assets'/folder/f'{prefix}_{local+1:05d}{ext}'
            bound=row['inputs'][role]
            if (root/Path(bound['path'])).resolve()!=path.resolve():raise ValueError('Producer input path mismatch: '+role)
            checked[role]=inspect_asset(path,role,bound['sha256'])
        outputs.append({'local_index':local,'source_frame_index':source_start+local,'inputs':checked})
    report={'status':'PASS_ALL65_INPUT_FILES_F32_CHANNEL_CONTRACT','full65_validated':True,
            'producer_root':str(root),'producer_ledger_path':str(ledger_path),'producer_ledger_sha256':sha(ledger_path),
            'width':4096,'height':2160,'fps':25,'frame_count':65,'duration_seconds':2.6,
            'source_frame_range':[source_start,source_start+64],'frames':outputs,'AE_executed':False,'visual_approval':False,'producer_method':ledger.get('method'),
            'color_contract':'Source RGB8 and F32 restored foreground are coded BT709, no EOTF; F has RGB only/noA. WhiteRGBA refined/raw prior carry independent F32 coverage A.',
            'next_requirement':'Later main authorization, actual native AE import/ALPHA readback and ZIP/FLOAT32 output pixel checks.'}
    write_new_json(in_package(args.receipt),report)
    print(json.dumps({'status':report['status'],'receipt':args.receipt}))

def config(args):
    validation=Path(args.validation).resolve(strict=True);report=read(validation)
    if report.get('status')!='PASS_ALL65_INPUT_FILES_F32_CHANNEL_CONTRACT' or report.get('full65_validated') is not True:
        raise ValueError('A fixture or incomplete validation cannot produce a host build config')
    if sha(report['producer_ledger_path'])!=report['producer_ledger_sha256']:raise ValueError('Producer ledger changed')
    for row in report['frames']:
        for item in row['inputs'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Validated asset changed before payload freeze')
    attempt=in_package(PACKAGE/'attempts'/args.attempt);attempt.mkdir(parents=True,exist_ok=False)
    root=Path(report['producer_root'])
    data={'host_build_authorized':bool(args.authorize_build),'full_input_validation_passed':True,
          'input_validation_path':str(validation),'input_validation_sha256':sha(validation),
          'refinement_method':report.get('producer_method'),
          'width':4096,'height':2160,'fps':25,'frame_count':65,
          'source_sequence_first':str(root/'assets/source/source_00001.png'),
          'foreground_sequence_first':str(root/'assets/foreground_rgb_float32/foreground_00001.exr'),
          'alpha_sequence_first':str(root/'assets/alpha_rgba_float32/alpha_00001.exr'),
          'raw_prior_sequence_first':str(root/'assets/raw_prior_rgba_float32/alpha_00001.exr'),
          'project_path':str(attempt/'LTX_BF16_4K_REFINED_ML_REVIEW.aep'),'receipt_path':str(attempt/'ae-build-receipt.json')}
    write_new_json(attempt/'ae-config.json',data)
    template=(Path(__file__).parent/'ae_import_4k_refined.jsx').read_text(encoding='utf-8')
    if template.count('__CONFIG_JSON__')!=1:raise ValueError('Expected one config token')
    code=template.replace('__CONFIG_JSON__',json.dumps(data,ensure_ascii=True,allow_nan=False))
    with (attempt/'ae-build.filled.jsx').open('x',encoding='utf-8') as stream:stream.write(code)
    print(json.dumps({'prepared_payload':str(attempt/'ae-build.filled.jsx'),'AE_executed':False,'host_build_authorized_flag':data['host_build_authorized']}))
def load_settings(path):
    path=Path(path).resolve();data=read(path)
    for key in ("delivery_root","producer_root"):
        data[key]=str((path.parent/Path(data[key])).resolve())
    return data

def main():
    global PACKAGE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--settings",default=str(Path(__file__).with_name("settings.example.json")))
    sub=p.add_subparsers(dest="command",required=True)
    v=sub.add_parser("verify-inputs");v.add_argument("--producer-root");v.add_argument("--receipt")
    c=sub.add_parser("prepare-config");c.add_argument("--validation");c.add_argument("--attempt");c.add_argument("--authorize-build",action="store_true")
    a=p.parse_args();settings=load_settings(a.settings);PACKAGE=Path(settings["delivery_root"])
    if a.command=="verify-inputs":
        a.producer_root=a.producer_root or settings["producer_root"];a.receipt=a.receipt or str(PACKAGE/"input-validation.json");a.source_start=int(settings["source_start_frame"]);verify(a)
    else:
        a.validation=a.validation or str(PACKAGE/"input-validation.json");a.attempt=a.attempt or settings["attempt"];config(a)
if __name__=="__main__":main()
