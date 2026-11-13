"""Loopback only; persistent prompt identity prevents automatic duplicate POST."""
from pathlib import Path
import ipaddress, json, time, uuid
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
from .common import read_json, save_json

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Loopback API redirect refused')

class LocalAPI:
    def __init__(self, url):
        parsed = urlparse(url)
        host = '127.0.0.1' if parsed.hostname == 'localhost' else parsed.hostname
        if (parsed.scheme != 'http' or parsed.username or parsed.password or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment or not host or not ipaddress.ip_address(host).is_loopback):
            raise ValueError('Use a plain loopback HTTP URL')
        self.url = 'http://' + ('[' + host + ']' if ':' in host else host) + ':' + str(parsed.port or 8188)
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + path, data=body,
                          headers={'Content-Type': 'application/json'} if body is not None else {})
        with self.opener.open(request, timeout=30) as response:
            data = response.read()
        return json.loads(data) if data else {}

def check_schema(api, prompt, required_flags=()):
    if required_flags:
        argv = api.request('/system_stats')['system'].get('argv', [])
        if not set(required_flags).issubset(argv):
            raise ValueError('Validated BF16/disk-offload server flags are missing')
    schemas = api.request('/object_info')
    for node in prompt.values():
        kind = node['class_type']
        schema = schemas.get(kind)
        if not schema:
            raise ValueError('Missing node: ' + kind)
        required = schema.get('input', {}).get('required', {})
        if set(required) - set(node['inputs']):
            raise ValueError(kind + ' is missing required inputs')
        for name, value in node['inputs'].items():
            spec = required.get(name) or schema.get('input', {}).get('optional', {}).get(name)
            if not spec or isinstance(value, list):
                continue
            options = spec[0] if isinstance(spec[0], list) else None
            if options and all(isinstance(x, str) for x in options) and value not in options:
                raise ValueError(kind + '.' + name + ' is not a registered option: ' + str(value))

def status(run, api):
    run = Path(run); state = read_json(run / 'state.json')
    intent = run / 'submission-intent.json'
    if 'prompt_id' not in state and intent.exists():
        state.update(read_json(intent)); state['phase'] = 'submission_uncertain'
        save_json(run / 'state.json', state)
    if 'prompt_id' in state:
        prompt_id = state['prompt_id']
        history = api.request('/history/' + prompt_id).get(prompt_id)
        if history:
            save_json(run / 'comfy-history.json', history)
            phase = history.get('status', {})
            if phase.get('status_str') == 'error' or any(m[0] == 'execution_error' for m in phase.get('messages', [])):
                state['phase'] = 'failed'
            elif phase.get('completed'):
                state['phase'] = 'completed'
        state['observed_queue'] = api.request('/queue')
        save_json(run / 'state.json', state)
    return state

def submit(run, api):
    run = Path(run); state = read_json(run / 'state.json')
    if state.get('prompt_id') or (run / 'submission-intent.json').exists():
        return status(run, api)  # Never POST a known or uncertain identity again.
    if state.get('phase') != 'prepared':
        raise ValueError('Only a fresh prepared run can submit')
    prompt = read_json(run / 'prompt_api.json')
    queue = api.request('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Queue busy; nothing submitted')
    identity = {'prompt_id': str(uuid.uuid4()), 'client_id': str(uuid.uuid4()), 'started_at': time.time()}
    try:
        with (run / 'submission-intent.json').open('x', encoding='utf-8') as stream:
            json.dump(identity, stream)
    except FileExistsError:
        return status(run, api)
    state.update(identity, phase='submitting'); save_json(run / 'state.json', state)
    payload = {'prompt': prompt, **identity}
    payload.pop('started_at')
    if (run / 'workflow_ui.json').exists():
        payload['extra_data'] = {'extra_pnginfo': {'workflow': read_json(run / 'workflow_ui.json')}}
    try:
        response = api.request('/prompt', payload)
        save_json(run / 'submission-response.json', response)
        if response.get('prompt_id') != identity['prompt_id']:
            raise RuntimeError('Unexpected server prompt identity; preserve response and investigate')
        state['phase'] = 'running'; save_json(run / 'state.json', state)
    except BaseException:
        state['phase'] = 'submission_uncertain'; save_json(run / 'state.json', state)
        raise
    return state
