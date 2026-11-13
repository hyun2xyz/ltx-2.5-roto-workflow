"""Small replay CLI for the validated65-frame25fps profile. No MCP or private skills."""
from pathlib import Path
import argparse, copy, json, shutil, time, uuid
from fractions import Fraction
from .common import REPO, command, config, frame_contract, read_json, save_json, sha
from .local_api import LocalAPI, check_schema, status, submit

def replace_values(value, replacements):
    if isinstance(value, dict):
        return {key: replace_values(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_values(item, replacements) for item in value]
    return replacements.get(value, value) if isinstance(value, str) else value

def prepare(cfg, source, run):
    source = Path(source).resolve(strict=True); run = Path(run).resolve()
    if run.exists():
        raise FileExistsError('Use a new run directory; existing runs are preserved')
    probe = json.loads(command([cfg['ffprobe'], '-v', 'error', '-show_streams', '-of', 'json', source], capture_output=True).stdout)
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    if ((stream['width'], stream['height']) != (4096, 2160) or Fraction(stream['avg_frame_rate']) != 25
            or stream.get('color_space') != 'bt709' or stream.get('color_range') != 'tv'):
        raise ValueError('This packaged profile requires4096x2160/25fps/BT709 limited source; adapt a copied profile explicitly')
    run.mkdir(parents=True)
    start = cfg['start_frame']; count = 65
    prefix = 'LTX_ROTO_' + uuid.uuid4().hex[:12]
    for folder in ('source', 'source_fhd'):
        (run / 'assets' / folder).mkdir(parents=True)
    trim = f'trim=start_frame={start}:end_frame={start+count},setpts=N/(25*TB),'
    color = ':in_color_matrix=bt709:in_range=tv:out_range=pc'
    native = trim + 'scale=iw:ih' + color + ',setsar=1,format=rgb24'
    fhd = trim + 'scale=1920:1088:force_original_aspect_ratio=decrease:flags=lanczos' + color + ',pad=1920:1088:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=rgb24'
    for folder, filt in [('source', native), ('source_fhd', fhd)]:
        command([cfg['ffmpeg'], '-v', 'error', '-n', '-i', source, '-map', '0:v:0', '-an', '-vf', filt,
                 '-frames:v', str(count), '-fps_mode', 'passthrough', run / 'assets' / folder / 'source_%05d.png'])
        if len(list((run / 'assets' / folder).glob('source_*.png'))) != count:
            raise ValueError('Requested source frames are missing; partial preparation preserved')
    reference = run / 'assets' / 'reference.mp4'
    command([cfg['ffmpeg'], '-v', 'error', '-n', '-framerate', '25', '-i', run / 'assets/source_fhd/source_%05d.png',
             '-frames:v', str(count), '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'fast', '-pix_fmt', 'rgb24', reference])
    replacements = {'__REFERENCE__': str(reference.resolve()), '__PREFIX__': prefix,
                    '__MATTE_PREFIX__': prefix + '/matte', '__FLOAT_PREFIX__': prefix + '/matte_float',
                    **{'__' + k.upper() + '__': v for k, v in cfg['registered_models'].items()}}
    prompt = replace_values(read_json(REPO / 'workflows/ltx_fusion65_api.json'), replacements)
    ui = replace_values(read_json(REPO / 'workflows/ltx_fusion65_ui.json'), replacements)
    check_schema(LocalAPI(cfg['comfy_url']), prompt, cfg['required_server_flags'])
    # Absolute reference is intentional: VHS_LoadVideoPath reads the local machine only.
    if prompt['5']['inputs']['video'] != str(reference.resolve()):
        raise ValueError('Reference path was not resolved')
    save_json(run / 'config.json', cfg); save_json(run / 'prompt_api.json', prompt); save_json(run / 'workflow_ui.json', ui)
    manifest = {'method': 'LTX25_BF16_FHD_FUSION33_EULER', 'source_original': str(source), 'source_sha256': sha(source),
                'source_start_frame': start, 'frame_count': 65, 'fps': 25, 'width': 1920, 'height': 1088,
                'original_grid': [4096,2160], 'prefix': prefix, 'reference_sha256': sha(reference),
                'source_decode': 'BT709limited YUV to fullRGB8 code values; no extra EOTF',
                'prior_fit': '1920x1013 fit, pad37top/38bottom; later remove y37:1050',
                'sampler': 'euler/nonancestral; temporal-only Fusion33, matching streamed33 guide, full spatial frame'}
    save_json(run / 'manifest.json', manifest); save_json(run / 'state.json', {'phase': 'prepared'})
    return {'prepared_only': True, 'run': str(run), 'frames': 65, 'GPU_started': False}

def collect(run):
    import numpy as np
    from .media import map_prior, read_exr, white_alpha, pixel_sha
    run = Path(run).resolve(); cfg = read_json(run / 'config.json'); manifest = read_json(run / 'manifest.json')
    ledger_path = run / 'prior-ledger.json'
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if not all(sha(item['path']) == item['sha256'] for row in ledger['frames'] for item in row['inputs'].values()):
            raise ValueError('Previously collected input changed')
        return ledger
    state = status(run, LocalAPI(cfg['comfy_url']))
    if state['phase'] != 'completed':
        raise RuntimeError('Owned Comfy prompt has not completed successfully')
    output_root = Path(cfg['comfy_output']).resolve(strict=True)
    stage = (output_root / manifest['prefix']).resolve(strict=True)
    if not stage.is_relative_to(output_root):
        raise ValueError('Comfy output prefix escapes configured root')
    files = sorted(stage.glob('matte_float_*.exr'))
    if len(files) != 65:
        raise ValueError('Expected exactly65 native F32 matte EXRs; preserve outputs and investigate')
    rows = []
    for local, origin in enumerate(files):
        index = local + 1
        raw = run / f'assets/raw_ltx_rgb/matte_{index:05d}.exr'; raw.parent.mkdir(parents=True, exist_ok=True)
        if raw.exists():
            if sha(raw) != sha(origin):
                raise ValueError('Existing raw EXR differs; refusing overwrite')
        else:
            shutil.copyfile(origin, raw)
        channels = read_exr(raw)
        if set(channels) != set('RGB') or channels['R'].shape != (1088,1920):
            raise ValueError('Native raw matte has wrong FLOAT32 channel/grid contract')
        alpha = np.clip((np.float32(.299)*channels['R'] + np.float32(.587)*channels['G']) + np.float32(.114)*channels['B'], 0, 1)
        mapped = map_prior(alpha)
        prior = run / f'assets/raw_prior_rgba_float32/alpha_{index:05d}.exr'
        if prior.exists():
            from .media import bit_exact
            if not bit_exact(read_exr(prior)['A'], mapped):
                raise ValueError('Existing mapped prior differs')
        else:
            white_alpha(prior, mapped)
        source = run / f'assets/source/source_{index:05d}.png'
        rows.append({**frame_contract(local, manifest['source_start_frame']), 'raw_prior_A_pixel_sha256': pixel_sha(mapped),
                     'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k,p in [('source',source),('raw_prior',prior),('raw_ltx_rgb',raw)]}})
    ledger = {'width':4096,'height':2160,'fps':25,'frame_count':65,'method':'Native4Ksource + mapped BF16 FHD raw LTX prior',
              'source_original': manifest['source_original'], 'source_sha256': manifest['source_sha256'], 'frames':rows}
    save_json(ledger_path,ledger)
    return {'status':'COLLECTED_65_NATIVE_PRIORS','ledger':str(ledger_path)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config',default='config/example.json'); p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('conditioning-prepare');p.add_argument('--config',default='config/example.json');p.add_argument('--out',required=True)
    p=sub.add_parser('conditioning-save');p.add_argument('--run',required=True);p.add_argument('--dest',required=True)
    for action in ('submit','status','collect','wait','release'):
        p=sub.add_parser(action);p.add_argument('--run',required=True)
        if action=='wait':p.add_argument('--timeout',type=int,default=21600)
    p=sub.add_parser('refine');p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--execution',choices=['isolated','batch'],default='isolated')
    p=sub.add_parser('export-rgba');p.add_argument('--producer',required=True);p.add_argument('--out',required=True);p.add_argument('--format',choices=['exr','png8'],default='exr')
    p=sub.add_parser('_worker');p.add_argument('phase',choices=['tile','foreground','pack']);p.add_argument('--out',required=True);p.add_argument('--local',type=int,required=True)
    p=sub.add_parser('_batch_worker');p.add_argument('phase',choices=['tile','foreground','pack']);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(config(args.config),args.source,args.out)
    elif args.action=='conditioning-prepare':
        from .conditioning import prepare_conditioning
        result=prepare_conditioning(config(args.config),args.out)
    elif args.action=='conditioning-save':
        from .conditioning import save_conditioning
        result=save_conditioning(args.run,args.dest)
    elif args.action=='release':
        from .conditioning import release
        result=release(args.run)
    elif args.action=='refine':
        from .refine import run_refine
        result=run_refine(args.run,args.out,args.execution)
    elif args.action=='export-rgba':
        from .export import export_rgba
        result=export_rgba(args.producer,args.out,args.format)
    elif args.action=='_batch_worker':
        from .refine import batch_worker
        result=batch_worker(args.phase,args.out)
    elif args.action=='_worker':
        from .refine import worker
        result=worker(args.phase,args.out,args.local)
    else:
        run=Path(args.run).resolve();cfg=read_json(run/'config.json');api=LocalAPI(cfg['comfy_url'])
        if args.action=='submit':
            check_schema(api,read_json(run/'prompt_api.json'),cfg['required_server_flags']);result=submit(run,api)
        elif args.action=='collect':result=collect(run)
        elif args.action=='status':result=status(run,api)
        else:
            deadline=time.monotonic()+args.timeout
            while True:
                result=status(run,api)
                if result['phase'] in ('completed','failed'):break
                if time.monotonic()>deadline:raise TimeoutError('Owned UUID may still be running; wait/status never resubmits')
                time.sleep(5)
    print(json.dumps(result,indent=2))
