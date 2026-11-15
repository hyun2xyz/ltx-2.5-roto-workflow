"""Optional shot-specific local corrections applied to straight foreground and alpha."""
from functools import lru_cache
from pathlib import Path

from .common import sha
from .media import read_exr, require_f32


def load_rules(recipe, ledger):
    rows = ledger['frames']
    if recipe['grid'] != [ledger['width'], ledger['height']]:
        raise ValueError('Repair recipe grid differs from the source')
    if recipe['source_start_frame'] != rows[0]['source_frame_index']:
        raise ValueError('Repair recipe source start frame differs from the ledger')
    if 'first_source_frame_sha256' in recipe:
        if recipe['first_source_frame_sha256'] != rows[0]['inputs']['source']['sha256']:
            raise ValueError('Repair recipe belongs to a different source clip')
    rules = recipe['rules']
    if not isinstance(rules, list) or not rules:
        raise ValueError('A repair recipe must contain at least one rule')
    for rule in rules:
        if rule['type'] not in ('hair-sky', 'jacket-edge'):
            raise ValueError(f"Unknown repair type: {rule['type']}")
        first, last = rule['frames']
        if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(rows):
            raise ValueError('Repair frames must be an inclusive zero-based range in the clip')
    return rules


class ShotRepairs:
    def __init__(self, recipe, ledger, producer):
        self.rules = load_rules(recipe, ledger)
        self.rows = ledger['frames']
        self.grid = (ledger['height'], ledger['width'])
        self.producer = Path(producer)
        self.reference_alpha = lru_cache(maxsize=2)(self._read_reference)

    def _input(self, local, key):
        item = self.rows[local]['inputs'][key]
        path = self.producer / item['path']
        if sha(path) != item['sha256']:
            raise ValueError(f'Changed {key} input at frame {local}')
        return path

    def _read_reference(self, local):
        if type(local) is not int or not 0 <= local < len(self.rows):
            raise ValueError('Reference frame outside the clip')
        alpha = read_exr(self._input(local, 'alpha'))['A']
        if alpha.shape != self.grid:
            raise ValueError('Reference matte grid differs from the source')
        alpha.setflags(write=False)
        return alpha

    def apply(self, local, channels):
        import numpy as np
        from PIL import Image

        active = [r for r in self.rules if r['frames'][0] <= local <= r['frames'][1]]
        if not active:
            return channels
        with Image.open(self._input(local, 'source')) as image:
            if image.mode != 'RGB' or image.size != self.grid[::-1]:
                raise ValueError('Repair input must be the original full-grid RGB8 frame')
            rgb = np.asarray(image)
        for rule in active:
            if rule['type'] == 'hair-sky':
                from .repair_hair import apply_hair
                channels = apply_hair(channels, rgb, local, rule)
            else:
                from .repair_edge import apply_edge
                channels = apply_edge(channels, rgb, local, rule, self.reference_alpha)
        if set(channels) != set('RGBA') or any(require_f32(v).shape != self.grid for v in channels.values()):
            raise ValueError('Repair changed the native grid or channel contract')
        if channels['A'].min() < 0 or channels['A'].max() > 1:
            raise ValueError('Repaired coverage outside [0, 1]')
        return channels
