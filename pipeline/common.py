from pathlib import Path
import hashlib, json, os, subprocess

REPO = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def command(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                          **kwargs)

def configured_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()

def config(path):
    result = read_json(path)
    for key in ('comfy_output', 'vitmatte_model_dir'):
        result[key] = str(configured_path(result[key]))
    return result

def frame_contract(local, start=75, fps=25):
    if not isinstance(local, int) or local < 0:
        raise ValueError('Expected nonnegative zero-based local frame')
    return {'local_index': local, 'source_frame_index': start + local,
            'file_index': local + 1, 'time_seconds': local / fps}
