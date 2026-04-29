from __future__ import annotations

import h5py  # type: ignore
import numpy as np
import os
import re
import shutil
from typing import Any, Literal
import torch
import yaml

from . import get_context


class IOInterface(object):
    h5: H5Interface
    output: OutputInterface
    yaml: YamlInterface

    def __init__(self):
        self.h5 = H5Interface()
        self.output = OutputInterface()
        self.yaml = YamlInterface()


class H5Interface(object):
    def __init__(self):
        pass

    def _open(self, file: str, mode: Literal['r', 'w', 'a']) -> h5py.File:
        return h5py.File(name=f'{file}.h5', mode=mode)

    def _open_read(self, file: str) -> h5py.File:
        return self._open(file, 'r')

    def _open_write(self, file: str) -> h5py.File:
        return self._open(file, 'a')

    def _read_dataset(self, file: h5py.File, tag: str) -> h5py.Dataset:
        _dset = file[tag]
        if not isinstance(_dset, h5py.Dataset):
            raise ValueError(f'The tag {tag} is not a dataset ({type(_dset)}).')
        return _dset

    def _read_group(self, file: h5py.File, tag: str) -> h5py.Group:
        _grp = file[tag]
        if not isinstance(_grp, h5py.Group):
            raise ValueError(f'The tag {tag} is not a group ({type(_grp)}).')
        return _grp

    def _require_group_recursive(self, file: h5py.File, *group: str) -> h5py.Group:
        _grp = file
        for grp in group:
            _grp = _grp.require_group(grp)  # type: ignore
        return _grp

    def read_tensor(self, path: str, tag: str, device: str = 'cuda') -> torch.Tensor:
        with self._open_read(path) as h5file:
            return torch.from_numpy(  # type: ignore
                np.array(self._read_dataset(h5file, tag))
            ).to(device=device)

    def write_tensor(self, path: str, tag: str, data: torch.Tensor, meta: dict[str, Any] = {}):
        with self._open_write(path) as h5file:
            _tag = tag.split('/')
            _grp = self._require_group_recursive(h5file, *_tag[:-1])
            if data.numel() == 1:
                try:
                    _dset = _grp.create_dataset(_tag[-1], data=data.cpu().numpy())  # type: ignore
                except ValueError:
                    del _grp[_tag[-1]]
                    _dset = _grp.create_dataset(_tag[-1], data=data.cpu().numpy())  # type: ignore
            else:
                try:
                    _dset = _grp.create_dataset(_tag[-1], data=data.cpu().numpy())  # type: ignore
                except ValueError:
                    del _grp[_tag[-1]]
                    _dset = _grp.create_dataset(_tag[-1], data=data.cpu().numpy())  # type: ignore
            _dset.attrs.update(meta)  # type: ignore

    def read_group_keys(self, path: str, tag: str) -> list[str]:
        with self._open_read(path) as h5file:
            _grp = self._read_group(h5file, tag)
            return list(_grp.keys())


class OutputInterface(object):
    def __init__(self):
        pass

    def clean_dir(self, path: str):
        shutil.rmtree(path)

    def make_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def make_or_clean_dir(self, path: str):
        if os.path.exists(path):
            self.clean_dir(path)
        self.make_dir(path)

    def list_times(self, path: str = 'output', range: tuple[float, float] | None = None) -> list[str]:
        _times = sorted([_d for _d in os.listdir(f'{path}') if re.search(r'\d+\.\d+', _d)],  key=float)

        if not _times:
            raise ValueError(f'No time directories found in \'{path}\'.')

        if range is not None:
            _times = [_d for _d in _times if range[0] <= float(_d) <= range[1]]

        return _times

    def list_runs(self, path: str) -> list[str]:
        _runs = sorted([f'{path}{_d}' for _d in os.listdir(f'{path}') if re.search(r'run_[0-9]*', _d)])

        if not _runs:
            raise ValueError(f'No run directories found in \'{path}\'.')

        return _runs


class YamlInterface(object):
    def __init__(self):
        pass

    def read(self, path: str) -> dict[str, Any]:
        with open(f'{path}.yaml', 'r') as file:
            return yaml.safe_load(file)

    def write(self, path: str, data: dict[str, Any]):
        with open(f'{path}.yaml', 'w') as file:
            yaml.dump(data, file, default_flow_style=False)


class UserConfiguration:
    _info: dict[str, Any]
    _file: str

    def __init__(self):
        self._info = get_context().io.yaml.read('config')
        self._add_base('grid/length', 2.0*torch.pi)
        self._add_base('numerics/device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self._add_base('numerics/precision', 'double')
        self._add_base('numerics/random_seed', torch.rand(1).item())

        if self.get('numerics/precision', str) == 'single':
            self._info['numerics']['precision'] = {}
            self._add_base('numerics/precision/complex', torch.complex64)
            self._add_base('numerics/precision/float', torch.float32)
            self._add_base('numerics/precision/int', torch.int32)
        elif self.get('numerics/precision', str) == 'double':
            self._info['numerics']['precision'] = {}
            self._add_base('numerics/precision/complex', torch.complex128)
            self._add_base('numerics/precision/float', torch.float64)
            self._add_base('numerics/precision/int', torch.int64)
        else:
            raise ValueError('Invalid numerics/precision entry in config file, expected \'single\' or \'double\'')

    def _add_base(self, key: str, value: Any):
        decomposed = key.split('/')
        sub = self._info
        for level in decomposed[:-1]:
            if level not in sub:
                sub[level] = {}
            sub = sub[level]

        if decomposed[-1] not in sub:
            sub[decomposed[-1]] = value

    def _get_recursive(self, sub: dict[str, Any], key: str) -> Any:
        try:
            return sub[key]
        except KeyError:
            return None

    def get(self, key: str, expected: type, default: Any = None) -> Any:
        sub = self._info
        path = '/'
        decomposed = key.split('/')
        for level in decomposed[:-1]:
            sub = self._get_recursive(sub, level)
            if sub is None:
                raise KeyError(f'Missing mapping \'{level}\' in \'{path}\'')
            path += f'{level}/'

        try:
            _value = sub[decomposed[-1]]
            if not isinstance(_value, expected):
                raise TypeError(f'Error reading \'{key}\' entry in config file, expected {expected}: {type(_value)}')
            return _value
        except KeyError:
            if default is None:
                raise KeyError(f'Missing \'{key}\' entry in config file')
            else:
                return default
