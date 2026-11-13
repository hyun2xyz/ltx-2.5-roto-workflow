"""Separate native Gemma BF16 encoding. No cached private conditioning is required."""
from pathlib import Path
import collections, json, shutil, struct, uuid
from .common import REPO, read_json, save_json, sha
from .local_api import LocalAPI, check_schema, status

def prepare_conditioning(cfg,run):
    run=Path(run).resolve()
    if run.exists():raise FileExistsError('Use a new conditioning stage directory')
    from .cli import replace_values
    prefix='LTX_CONDITIONING_'+uuid.uuid4().hex[:12]
    prompt=replace_values(read_json(REPO/'workflows/empty_conditioning_api.json'),
                          {'__TEXT_ENCODER__':cfg['registered_models']['text_encoder'],'__CONDITIONING_PREFIX__':prefix+'/empty_prompt'})
    check_schema(LocalAPI(cfg['comfy_url']),prompt,cfg['required_server_flags'])
    run.mkdir(parents=True)
    save_json(run/'config.json',cfg);save_json(run/'prompt_api.json',prompt)
    save_json(run/'manifest.json',{'method':'GEMMA4_NATIVE_BF16_EMPTY_CONDITIONING','prefix':prefix,'text':'',
                                  'encoder':cfg['registered_models']['text_encoder'],'dtype':'Preserve native Comfy conditioning and options; no conversion'})
    save_json(run/'state.json',{'phase':'prepared'})
    return {'prepared_only':True,'run':str(run),'nodes':3,'GPU_started':False}

def save_conditioning(run,destination):
    run=Path(run).resolve();cfg=read_json(run/'config.json');manifest=read_json(run/'manifest.json')
    if manifest['method']!='GEMMA4_NATIVE_BF16_EMPTY_CONDITIONING':raise ValueError('Expected an owned conditioning stage')
    if status(run,LocalAPI(cfg['comfy_url']))['phase']!='completed':raise RuntimeError('Owned empty-conditioning prompt is not complete')
    output=Path(cfg['comfy_output']).resolve(strict=True);stage=(output/manifest['prefix']).resolve(strict=True)
    if not stage.is_relative_to(output):raise ValueError('Conditioning staging escaped output root')
    files=list(stage.glob('empty_prompt_*.safetensors'))
    if len(files)!=1:raise ValueError('Expected one native saved conditioning file')
    source=files[0];destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>16*1024**2:raise ValueError('Unexpected safetensors header size')
        header=json.loads(stream.read(length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    if 'conditioning' not in tensors:raise ValueError('Native conditioning tensor missing')
    if destination.exists():
        if sha(destination)!=sha(source):raise FileExistsError('Different saved conditioning exists; preserve it')
    else:shutil.copyfile(source,destination)
    if sha(destination)!=sha(source):raise ValueError('Conditioning copy changed')
    result={'status':'NATIVE_CONDITIONING_SAVED','path':str(destination),'sha256':sha(destination),
            'tensor_dtypes':dict(collections.Counter(v['dtype'] for v in tensors.values())),
            'conditioning_dtype':tensors['conditioning']['dtype'],'all_tensor_and_option_bytes_preserved':True,
            'next':'Run explicit release on this idle dedicated server, register this destination under embeddings, update config conditioning name.'}
    save_json(run/'conditioning-receipt.json',result);return result

def release(run):
    run=Path(run).resolve();cfg=read_json(run/'config.json');api=LocalAPI(cfg['comfy_url'])
    if not cfg.get('dedicated_owned_server') or api.url.endswith(':8188'):
        raise RuntimeError('Release is restricted to an explicitly dedicated owned server; primary8188 is protected')
    state=status(run,api)
    if state.get('phase')!='completed':raise RuntimeError('Owned prompt must be successfully complete before release')
    queue=api.request('/queue')
    if queue['queue_running'] or queue['queue_pending']:raise RuntimeError('Queue not idle; no release performed')
    response=api.request('/free',{'unload_models':True,'free_memory':True})
    result={'status':'OWNED_IDLE_SERVER_RELEASE_REQUESTED','url':api.url,'prompt_id':state['prompt_id'],'response':response,
            'note':'Server stays running. This is a cache-release request, not a measured free-VRAM guarantee.'}
    save_json(run/'release-receipt.json',result);return result
