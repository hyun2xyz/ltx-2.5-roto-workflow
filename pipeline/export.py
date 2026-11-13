"""Direct Python delivery: straight RGBA, with no AE and no model execution."""
from pathlib import Path
import math
from .common import read_json, save_json, sha
from .media import read_exr, write_exr, require_f32


def straight_channels(foreground, matte):
    import numpy as np
    if set(foreground) != set('RGB') or set(matte) != set('RGBA'):
        raise ValueError('Expected RGB-only F and separate white RGBA coverage')
    alpha = require_f32(matte['A'])
    if alpha.ndim != 2 or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError('Invalid coverage')
    for c in 'RGB':
        require_f32(foreground[c]); require_f32(matte[c])
        if foreground[c].shape != alpha.shape or matte[c].shape != alpha.shape:
            raise ValueError('Foreground/matte grids differ')
        if not np.all(matte[c] == 1):
            raise ValueError('Use the coverage A channel, not white matte RGB luma')
    # Do NOT multiply F by A here. The resulting file has straight alpha.
    return {**{c: foreground[c] for c in 'RGB'}, 'A': alpha}


def export_rgba(producer, output, kind='exr'):
    import numpy as np
    if kind not in ('exr', 'png8'):
        raise ValueError('Choose EXR F32 master or PNG8 review')
    producer = Path(producer).resolve(strict=True)
    ledger_path = producer/'frame-ledger.json'
    ledger = read_json(ledger_path)
    rows = ledger['frames']; fps = float(ledger['fps'])
    if not rows or len(rows) != ledger['frame_count'] or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid frame clock')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    origin = rows[0]['source_frame_index']; result = []
    for local, row in enumerate(rows):
        if row['local_index'] != local or row['source_frame_index'] != origin+local:
            raise ValueError('Noncontiguous source clock')
        if 'time_seconds' in row and not math.isclose(row['time_seconds'], local/fps, abs_tol=1e-9):
            raise ValueError('Frame timestamp mismatch')
        planes = {}
        for role in ('foreground', 'alpha'):
            item = row['inputs'][role]; path = producer/Path(item['path'])
            if sha(path) != item['sha256']:
                raise ValueError('Changed input: '+role)
            planes[role] = read_exr(path)
        channels = straight_channels(planes['foreground'], planes['alpha'])
        if channels['A'].shape != (ledger['height'], ledger['width']):
            raise ValueError('Grid differs from ledger')
        target = output/f'rgba_{local+1:05d}.{ "exr" if kind == "exr" else "png" }'
        if kind == 'exr':
            write_exr(target, channels, 'BT709-code-values')
        else:
            from PIL import Image
            values = np.stack([channels[c] for c in 'RGBA'], axis=2)
            quantized = np.floor(np.clip(values, 0, 1)*255+.5).astype(np.uint8)
            Image.fromarray(quantized).save(target)
            with Image.open(target) as image:
                if not np.array_equal(np.asarray(image), quantized):
                    raise ValueError('PNG review readback mismatch')
        result.append({'local_index': local, 'source_frame_index': origin+local,
                       'time_seconds': local/fps, 'path': str(target), 'sha256': sha(target)})
    report = {'status': 'EXPORTED', 'format': kind, 'alpha_association': 'straight',
              'RGB': 'restored BT709 code values; no EOTF, no extra alpha multiplication',
              'precision': 'native FLOAT32, bit-exact write/readback' if kind == 'exr' else '8-bit clamped/rounded review; not F32 master',
              'width': ledger['width'], 'height': ledger['height'], 'fps': fps,
              'frame_count': len(rows), 'duration_seconds': len(rows)/fps,
              'input_ledger_sha256': sha(ledger_path), 'frames': result,
              'AE_used': False, 'inference_started': False}
    save_json(output/'export-receipt.json', report)
    return {'status': report['status'], 'frames': len(rows), 'format': kind,
            'receipt': str(output/'export-receipt.json')}
