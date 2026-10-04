"""Removing MTP must not silently change capacity or short-prefill settings."""
import json
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / 'scripts/run_dsv41_dspark_4090_candidate.sh'


def capture(tmp_path, enabled, module_loading=None):
    interpreter = tmp_path / 'capture-argv'
    interpreter.write_text('#!/usr/bin/python3\nimport json, os, sys\n'
        'print(json.dumps({"args": sys.argv[1:], "graphs": '
        'os.environ["FREETOKEN_DSV41_DSPARK_CUDA_GRAPH_LENGTHS"], '
        '"draft_graph": os.environ["FREETOKEN_DSV41_DSPARK_DRAFT_CUDA_GRAPH"], '
        '"module_loading": os.environ["CUDA_MODULE_LOADING"]}))\n')
    interpreter.chmod(0o700)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('DSPARK_', 'FREETOKEN_DSV41_'))
           and key != 'FREETOKEN_CUDA_MODULE_LOADING'}
    env.update(FREETOKEN_PYTHON=str(interpreter), MODEL_PATH='/synthetic/model',
               CUDA_VISIBLE_DEVICES='synthetic-devices', DSPARK_SPARSE_PREFILL_MAX_TOKENS='256')
    if enabled is not None:
        env['DSPARK_ENABLED'] = enabled
    if module_loading is not None:
        env['FREETOKEN_CUDA_MODULE_LOADING'] = module_loading
    return subprocess.run(['bash', str(LAUNCHER)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize('enabled', [None, '1', '0'])
def test_mtp_toggle_preserves_common_profile(tmp_path, enabled):
    result = capture(tmp_path, enabled)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record['module_loading'] == 'EAGER'
    args = record['args']
    for flag, value in {'--max-seq-len-override': '524288', '--max-prefill-length': '8192',
                        '--moe-cache-sizes': '256,1472,2350', '--dsv41-expert-shards': '32,160,192',
                        '--dsv41-prefill-expert-shards': '128,160,96', '--vision-device': '3',
                        '--moe-sparse-prefill-max-tokens': '256', '--moe-backend': 'offload',
                        '--moe-cpu-threads': '0', '--max-running-requests': '1'}.items():
        assert args[args.index(flag) + 1] == value
    assert '--moe-prefill-hit-d2d' in args
    if enabled == '0':
        assert not any(flag.startswith(('--speculative', '--dspark')) for flag in args)
        assert record['graphs'] == '' and record['draft_graph'] == '0'
    else:
        assert '--speculative-dspark' in args
        assert args[args.index('--dspark-verification-length') + 1] == '4'
        assert record['graphs'] == '4' and record['draft_graph'] == '1'


def test_mtp_toggle_rejects_invalid_setting(tmp_path):
    result = capture(tmp_path, 'false')
    assert result.returncode == 2 and result.stdout == ''
    assert 'must be 0 or 1' in result.stderr


@pytest.mark.parametrize('mode', ['EAGER', 'LAZY'])
def test_module_loading_can_be_isolated_without_changing_common_profile(tmp_path, mode):
    result = capture(tmp_path, '0', mode)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record['module_loading'] == mode
    assert '--speculative-dspark' not in record['args']
    assert record['args'][record['args'].index('--moe-sparse-prefill-max-tokens') + 1] == '256'


def test_module_loading_rejects_unknown_mode(tmp_path):
    result = capture(tmp_path, '0', 'OFF')
    assert result.returncode == 2 and result.stdout == ''
    assert 'must be EAGER or LAZY' in result.stderr
