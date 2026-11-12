"""Configuration, atomic output, logging and reproducibility helpers."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
if (ROOT / 'deps').is_dir():
    sys.path.insert(0, str(ROOT / 'deps'))

import numpy as np
import torch


def read_config(path=ROOT / 'config.json'):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if config['system'] not in ('IEEE_14', 'IEEE_118', 'IEEE_300', 'IEEE_1354'):
        raise ValueError('system must be IEEE_14, IEEE_118, IEEE_300 or IEEE_1354')
    if config['mode'] not in ('train_test', 'test', 'prepare', 'check'):
        raise ValueError('mode must be train_test, test, prepare or check')
    if config.get('data',{}).get('source')!='csv':raise ValueError('CSV-only source required')
    for name in ('batch_size', 'evaluation_batch_size', 'min_batch_size', 'cpu_threads'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['min_batch_size'] > config['batch_size']:
        raise ValueError('min_batch_size cannot exceed batch_size')
    training = config['training']
    for name in ('epochs', 'initializer_width', 'validate_every', 'operator_passes'):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f'training.{name} must be a positive integer')
    if training['learning_rate'] <= 0 or training['operator_noise_std'] <= 0 or training['operator_ridge'] < 0:
        raise ValueError('Invalid learning rate, noise standard deviation or ridge')
    inference = config['inference']
    if type(inference['max_blocks']) is not int or not 1 <= inference['max_blocks'] <= 10:
        raise ValueError('max_blocks must be an integer in [1,10]; one block has six NN calls')
    if inference['tolerance'] != 1e-4:
        raise ValueError('This experiment uses the fixed strict max PF mismatch <1e-4 criterion')
    if not inference['momentum_candidates'] or any(not 0 <= b < 1 for b in inference['momentum_candidates']):
        raise ValueError('Momentum candidates must be in [0,1)')
    if inference['momentum_start_call'] < 1:
        raise ValueError('Initializer must not enter momentum history')
    if config['benchmark']['single_samples'] < 1 or config['benchmark']['repeats'] < 1:
        raise ValueError('Benchmark sample count and repetitions must be positive')
    return config


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Logger:
    def __init__(self, directory, mode):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('train.log' if mode == 'train_test' else mode + '.log')
        self.run_id = time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}'

    def __call__(self, event, **fields):
        value = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'run_id': self.run_id,
                 'event': event, **fields}
        text = json.dumps(value, ensure_ascii=False)
        print(text, flush=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(text + '\n')


def setup(config):
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    if config['mode'] != 'prepare' and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; set device to cpu or use the configured Python environment')
    return device


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    return {p.name: sha256(p) for p in sorted(ROOT.glob('*.py'))}


def model_meta(meta, config):
    value = copy.deepcopy(meta)
    value.update(include_state=config.get('include_state',False),initializer_width=config['training']['initializer_width'],
                 momentum_beta=config['inference']['momentum_candidates'][0], momentum_start_call=config['inference']['momentum_start_call'])
    return value
