"""Shared tested native-F32 helpers; the active100-export checker is v2."""
from pathlib import Path
import json
import numpy as np
from media_precision import read_exr,sha

A_TOL = 2 ** -22

RGB_TOL = 2 ** -22

COMPOSITE_TOL = 2 ** -20

PATTERNS = {'source':'native-source/source_%05d.exr',
            'foreground':'native-foreground/foreground_%05d.exr',
            'raw_alpha':'native-raw-alpha/rawprior_%05d.exr',
            'refined_alpha':'native-refined-alpha/refined_%05d.exr',
            'raw_roto':'native-raw-roto/raw_%05d.exr',
            'source_refined':'native-source-refined/source_refined_%05d.exr',
            'clean':'native-clean/clean_%05d.exr',
            'composite':'native-composite/composite_%05d.exr'}

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def check(error,tolerance,label):
    if not np.isfinite(error) or error>tolerance:raise ValueError(f'{label}: {error} exceeds {tolerance}')
    return float(error)

def difference(actual,expected):return float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))

def native(path):
    values,header=read_exr(path,alpha=True)
    if values.shape!=(2160,4096,4) or set(header['channel_pixel_types'])!=set('RGBA'):
        raise ValueError('Native4K RGBA channel/grid mismatch')
    if header['compression_code']!=3 or not header['all_channels_float32']:
        raise ValueError('Native ZIP/FLOAT32 channel storage proof required')
    return values,{'path':str(path),'sha256':sha(path),'header':header}
