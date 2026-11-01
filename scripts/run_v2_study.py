"""Serialize the predeclared V2 pilots, fresh training, dev selection and new holdout.

Windows owner-PC entry point. Never opens the original official test. All GPU
work uses the unchanged WSL environment. Source and scientific choices must be
committed before the corresponding jobs. Existing outputs are not overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES
from scripts.complete_technical_pipeline import (commit_new_paths, ensure_clean,
    parse_job_record, path_inside, read_json, relative, verify_run_inputs, verify_run_outputs,
    windows_to_wsl, write_new, write_state)
from scripts.run_job import git

FAMILIES = ('bce', 'focal', 'structured')
GRID = [.005,.01,.02,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95]
BIAS_GRID = [-8,-6,-4,-2,0,2,4,6,8]
DATA = ('data/processed/train.jsonl', 'data/processed/validation.jsonl',
        'data/splits/split_manifest_v2.csv', 'data/splits/split_summary_v2.json')
REVISION = '8949b909ec900327062f0ebf497f51aef5e6f0c8'
PRETRAINED_SHA256 = '340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9'
STUDY_ID = 'ragtruth-v2-fresh-original-train-holdout'


def require(value, message):
    if not value:
        raise ValueError(message)


def metric_key(artifact):
    require(artifact.get('schema_version') == 1 and artifact.get('status') == 'selected_on_validation'
            and artifact.get('validation_split') == 'validation', 'decoder was not selected on dev')
    require(artifact.get('label_types') == list(LABEL_TYPES), 'decoder label order mismatch')
    scores = artifact['validation_metrics']
    for score in [scores['macro_f1_types'], scores['any']['f1'],
                  *(scores[label]['f1'] for label in LABEL_TYPES)]:
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                'invalid validation F1')
    require(math.isclose(scores['macro_f1_types'], sum(scores[label]['f1'] for label in LABEL_TYPES) / len(LABEL_TYPES),
                         rel_tol=1e-12, abs_tol=1e-12), 'typed macro F1 disagrees with fixed four classes')
    return scores['macro_f1_types'], scores['any']['f1']


def select_family(artifacts):
    require(set(artifacts) <= set(FAMILIES) and {'bce', 'focal'} <= set(artifacts), 'missing required model family')
    return max(artifacts, key=lambda name: (*metric_key(artifacts[name]), -FAMILIES.index(name)))


def wsl_git_environment(root, git_dir, original):
    """Bypass the Windows .git pointer when Linux Git opens this worktree."""
    root, git_dir, original = map(PureWindowsPath, (root, git_dir, original))
    require(all(path.is_absolute() and '..' not in path.parts for path in (root,git_dir,original)),
            'Git environment requires absolute nontraversing Windows paths')
    common = original / '.git'
    worktrees = common / 'worktrees'
    require(root != original and root.is_relative_to(original), 'V2 worktree must be inside original project')
    require(git_dir != worktrees and git_dir.is_relative_to(worktrees),
            'worktree Git directory must be inside original .git/worktrees')
    return ['GIT_DIR='+windows_to_wsl(git_dir), 'GIT_COMMON_DIR='+windows_to_wsl(common),
            'GIT_WORK_TREE='+windows_to_wsl(root)]


def pilot_resources(summary):
    row = summary['epochs'][0]
    durations = [row[side]['elapsed_seconds'] for side in ('train','validation')]
    require(all(type(value) in (int,float) and math.isfinite(value) and value > 0 for value in durations),
            'invalid pilot elapsed time')
    peaks = [row[side].get('peak_allocated_bytes') for side in ('train','validation')]
    require(all(value is None or (type(value) in (int,float) and math.isfinite(value) and value >= 0)
                for value in peaks), 'invalid pilot peak memory')
    estimate = 3 * (durations[0] / 32 * 11431 + durations[1] / 16 * 2240)
    peak = max((value for value in peaks if value is not None), default=None)
    require(math.isfinite(estimate), 'invalid pilot full-run estimate')
    reason = ('estimated three-epoch run exceeds 18000 seconds' if estimate > 18000 else
              'peak allocation exceeds 14 GiB' if peak is not None and peak > 14 * 1024**3 else None)
    return {'estimated_full_three_epoch_seconds':estimate, 'peak_allocated_bytes':peak,
            'structured_cancellation_reason':reason,
            'estimate_note':'linear pilot response-rate extrapolation; excludes checkpoint and startup overhead'}


@contextmanager
def original_queue_lock(original):
    """Reserve the original GPU queue while child jobs use the V2 queue."""
    lock = original / 'outputs/.run.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    token = {'pid': os.getpid(), 'scope': 'ragtruth-v2-study',
             'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        with lock.open('x', encoding='utf-8') as handle:
            json.dump(token, handle)
    except FileExistsError as exc:
        raise ValueError('another original project job is running') from exc
    try:
        yield
    finally:
        # Never unlink a lock replaced by another process or an operator.
        try:
            if read_json(lock) == token:
                lock.unlink()
        except (OSError, ValueError):
            pass


class Study:
    def __init__(self, output):
        self.root = ROOT.resolve()
        require(self.root.name == 'ragtruth-v2' and self.root.parent.name == '.cache', 'use isolated V2 worktree')
        ensure_clean(self.root)
        self.original = self.root.parent.parent
        self.cpu = self.original / '.venv/Scripts/python.exe'
        require(self.cpu.is_file(), 'owner CPU interpreter missing')
        self.output = path_inside(self.root, output, exists=False)
        require(self.output != self.root / 'outputs' and self.output.is_relative_to(self.root / 'outputs')
                and self.output.suffix == '.json', 'output must be a JSON file inside outputs/')
        self.status = self.output.with_suffix('.status.json')
        self.work = self.output.parent / (self.output.stem + '_work')
        require(not any(p.exists() for p in (self.output, self.status, self.work)), 'existing study output')
        self.work.mkdir(parents=True)
        self.state = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
                      'study':'fresh V2 internal holdout; original official test excluded',
                      'original_official_test_read':False, 'stages':{}, 'canceled_families':{}}
        self.initial_inputs = self.inputs()
        self.input_snapshot = {p:sha256_file(path_inside(self.root,p)) for p in self.initial_inputs}
        self.state['initial_inputs_sha256'] = dict(self.input_snapshot)
        self.data_identity = {'split_summary':self.input_snapshot[DATA[3]],
                              'split_manifest':self.input_snapshot[DATA[2]],
                              'train.jsonl':self.input_snapshot[DATA[0]],
                              'validation.jsonl':self.input_snapshot[DATA[1]]}
        split_summary = read_json(self.root / DATA[3])
        require(split_summary.get('study_id') == STUDY_ID and split_summary.get('seed') == 20261101
                and split_summary.get('original_official_test_annotations_read') is False, 'wrong V2 data cohort')
        require(all(split_summary['processed_sha256'][name] == self.data_identity[name]
                    for name in ('train.jsonl','validation.jsonl')), 'V2 data summary checksum mismatch')

    def verify_source_snapshot(self):
        require(self.inputs() == self.initial_inputs, 'study source/config input set changed')
        for name, digest in self.input_snapshot.items():
            require(sha256_file(path_inside(self.root,name)) == digest, 'study input changed: ' + name)

    def progress(self, phase, **details):
        self.state['phase'] = phase
        self.state['stages'][phase] = details
        write_state(self.status, self.state)
        print(json.dumps({'phase':phase, **details}), flush=True)

    def inputs(self):
        tracked = git(self.root, 'ls-files').decode('utf-8').splitlines()
        names = [p for p in tracked if p.endswith('.py') or
                 (p.startswith('configs/') and p.endswith('.json') and not p.startswith('configs/v2/final/')) or
                 (Path(p).name.startswith('requirements') and p.endswith('.txt'))]
        return sorted(set(names) | set(DATA) | {'AGENTS.md','docs/V2_PLAN.md'})

    def files(self, directory):
        require(directory.is_dir() and directory.resolve().is_relative_to(self.root), 'missing or external artifact directory')
        return [relative(self.root, p) for p in sorted(directory.rglob('*')) if p.is_file()]

    def execute(self, command, name):
        self.progress(name)
        child = subprocess.run(command, cwd=self.root, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
        (self.work / (name + '.log')).write_text(child.stdout + child.stderr, encoding='utf-8')
        require(child.returncode == 0, name + ' failed; see study log')
        return child.stdout

    def job(self, name, command, outputs, *, stage='predict', split='validation',
            inputs=(), gpu=True, timeout=21600, frozen=False):
        self.verify_source_snapshot()
        require(split != 'test' or frozen, 'holdout job requires a frozen protocol')
        declared = sorted(set(self.initial_inputs) | set(inputs))
        for name_ in declared:
            require(path_inside(self.root,name_).is_file(), 'job input must be a file: ' + name_)
        config = {'schema_version':1, 'name':'v2-' + name.replace('_','-'), 'stage':stage,
                  'split':split, 'seed':42, 'inputs':declared,
                  'command':command, 'expected_outputs':outputs, 'timeout_seconds':timeout}
        if frozen:
            destination = self.root / 'configs/v2/final'
            path = destination / (name + '.json')
            protocol = destination / (name + '.protocol.json')
            config['frozen_protocol'] = relative(self.root, protocol)
            write_new(path, config)
            write_new(protocol, {'status':'frozen', 'scientific_decision':'docs/V2_FINAL_PROTOCOL.json',
                'scientific_decision_sha256':sha256_file(self.root / 'docs/V2_FINAL_PROTOCOL.json'),
                'test_config_sha256':sha256_file(path),
                'inputs_sha256':{p:sha256_file(self.root / p) for p in config['inputs']}})
            commit_new_paths(self.root, [path,protocol], 'Freeze V2 ' + name + ' byte identities')
        else:
            path = self.work / (name + '.json')
            write_new(path, config)
        if gpu:
            git_dir = git(self.root,'rev-parse','--absolute-git-dir').decode('utf-8').strip()
            git_environment = wsl_git_environment(str(self.root),git_dir,str(self.original))
            argv = ['wsl.exe','--distribution','Ubuntu-24.04','--user','serda','--cd',windows_to_wsl(str(self.root)),
                '--','env','HSA_ENABLE_DXG_DETECTION=1','HF_HUB_CACHE=.cache/hf-hub','HF_HUB_OFFLINE=1',
                'TRANSFORMERS_OFFLINE=1',*git_environment,
                '/opt/banking77-venv/bin/python','scripts/run_job.py',relative(self.root,path)]
        else:
            argv = [str(self.cpu),'scripts/run_job.py',relative(self.root,path)]
        record_path = parse_job_record(self.execute(argv,name), root=self.root)
        record = verify_run_inputs(read_json(record_path), root=self.root)
        verify_run_outputs(record_path.parent, record)
        require(record.get('config') == config, 'job record config differs from requested job')
        require(all(record['inputs_sha256'].get(p) == digest for p,digest in self.input_snapshot.items()),
                'job did not use the initial frozen study inputs')
        self.verify_source_snapshot()
        self.state['stages'][name]['record'] = relative(self.root,record_path)
        return record_path.parent

    def train(self, family, *, pilot=False):
        require(family in FAMILIES, 'unknown model family')
        script = 'scripts/structured_transformer.py' if family == 'structured' else 'scripts/transformer.py'
        summary = 'structured_summary.json' if family == 'structured' else 'transformer_summary.json'
        command = ['{python}',script,'train','--config',f'configs/v2/{family}.json',
                   '--device','cuda','--output','{run_dir}']
        if pilot:
            command += ['--limit-train','32','--limit-validation','16','--epochs','1']
        snapshot = self.root / ('.cache/hf-hub/models--answerdotai--ModernBERT-base/snapshots/' + REVISION)
        files = self.files(snapshot)
        require(all((snapshot/name).is_file() for name in ('config.json','model.safetensors','tokenizer.json',
                    'tokenizer_config.json','special_tokens_map.json')), 'incomplete pinned pretrained snapshot')
        require(sha256_file(snapshot/'model.safetensors') == PRETRAINED_SHA256, 'wrong original pretrained weights')
        run = self.job(family + ('_pilot' if pilot else '_train'), command, [summary], stage='pilot' if pilot else 'train',
                       split='train', inputs=files, timeout=900 if pilot else 21600)
        data = read_json(run / summary)
        require(data.get('status') == 'complete', 'incomplete ' + family + ' training')
        require(len(data['epochs']) == (1 if pilot else 3), 'unexpected epoch count')
        config = data['config']
        require(config['revision'] == REVISION and config.get('seed') == 42, 'wrong pretrained revision or seed')
        require(config.get('split_manifest') == DATA[2] and config.get('split_summary') == DATA[3]
                and config.get('processed_dir') == 'data/processed', 'training used a different data cohort')
        require(data.get('counts') == ({'train':32,'validation':16} if pilot else {'train':11431,'validation':2240})
                and data.get('limits') == ({'train':32,'validation':16} if pilot else {'train':None,'validation':None}),
                'training counts or limits differ from the full V2 protocol')
        require(data.get('label_types') == list(LABEL_TYPES) and data.get('data_sha256') == self.data_identity,
                'training data or label identity mismatch')
        checkpoints = [run / f'checkpoints/epoch-{i}' for i in range(1, (1 if pilot else 3) + 1)]
        require(all((p/'checkpoint.json').is_file() for p in checkpoints), 'missing declared epoch checkpoint')
        require(data.get('saved_checkpoints') == [p.relative_to(run).as_posix() for p in checkpoints],
                'training did not save every declared epoch')
        for epoch, checkpoint in enumerate(checkpoints, 1):
            saved = read_json(checkpoint/'checkpoint.json')
            require(saved.get('label_types') == list(LABEL_TYPES) and saved.get('config') == config
                    and saved.get('metadata',{}).get('data_sha256') == self.data_identity
                    and saved['metadata'].get('epoch') == epoch, 'saved epoch checkpoint identity mismatch')
        return run, data, checkpoints

    def calibrate(self, family, checkpoints):
        inputs = [p for ckpt in checkpoints for p in self.files(ckpt)]
        if family == 'structured':
            config = {'split':'validation','checkpoints':[relative(self.root,p) for p in checkpoints],
                      'o_bias_grid':BIAS_GRID}
            script = 'scripts/structured_transformer.py'
            command = ['{python}',script,'calibrate']
        else:
            config = {'schema_version':1,'split':'validation','selection_metric':'macro_f1_types',
                      'checkpoints':[relative(self.root,p) for p in checkpoints],'threshold_grid':GRID}
            command = ['{python}','scripts/calibrate_transformer.py']
        path = self.work / (family + '_calibration_config.json')
        write_new(path,config)
        command += ['--config',relative(self.root,path),'--device','cuda','--output','{run_dir}']
        run = self.job(family + '_calibration',command,
                        ['calibration.json','calibration_report.json','validation_predictions.jsonl'],
                        inputs=[*inputs,relative(self.root,path)])
        decoder = self.validate_decoder(family,run/'calibration.json')
        require(decoder['checkpoint'] in [relative(self.root,p) for p in checkpoints],
                'calibration selected an undeclared checkpoint')
        return run

    def validate_decoder(self, family, path):
        artifact = read_json(path)
        metric_key(artifact)
        require(artifact.get('selection_metric') == 'macro_f1_types', 'wrong decoder selection metric')
        require(artifact.get('official_test_evaluated',False) is False
                and artifact.get('holdout_evaluated',False) is False, 'decoder has seen a holdout')
        require(artifact.get('data_sha256') == self.data_identity, 'decoder used a different data cohort')
        checkpoint = path_inside(self.root,artifact['checkpoint'])
        require(checkpoint.is_relative_to(self.root/'outputs/runs') and checkpoint.is_dir(),
                'selected checkpoint must belong to this V2 study')
        saved_path = checkpoint/'checkpoint.json'
        require(sha256_file(saved_path) == artifact.get('checkpoint_sha256'), 'decoder checkpoint checksum mismatch')
        saved = read_json(saved_path)
        require(saved.get('label_types') == list(LABEL_TYPES)
                and saved.get('metadata',{}).get('data_sha256') == self.data_identity,
                'selected checkpoint data/label identity mismatch')
        require(saved.get('config',{}).get('revision') == REVISION, 'selected checkpoint pretrained revision mismatch')
        expected = saved.get('files_sha256',{})
        actual = {p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob('*')
                  if p.is_file() and p != saved_path}
        require(expected and set(expected) == actual, 'selected checkpoint file manifest mismatch')
        for name,digest in expected.items():
            require(sha256_file(path_inside(checkpoint,name)) == digest, 'selected checkpoint file checksum mismatch')
        if family == 'structured':
            require(artifact.get('o_bias_grid') == BIAS_GRID
                    and set(artifact.get('o_bias_by_type',{})) == set(LABEL_TYPES)
                    and all(type(v) in (int,float) and v in BIAS_GRID
                            for v in artifact['o_bias_by_type'].values()), 'wrong structured decoder grid')
        else:
            require(artifact.get('threshold_grid') == GRID
                    and set(artifact.get('thresholds_by_type',{})) == set(LABEL_TYPES)
                    and all(type(v) in (int,float) and v in GRID
                            for v in artifact['thresholds_by_type'].values()), 'wrong sigmoid decoder grid')
            require(saved['config'].get('loss','bce') == family, 'selected checkpoint loss/family mismatch')
        return artifact

    def compare(self, predictions, *, test=False):
        command = ['{python}','evaluate.py','--split','test' if test else 'validation',
                   '--gold','data/processed/test.jsonl' if test else 'data/processed/validation.jsonl',
                   '--manifest','data/splits/split_manifest_v2.csv','--split-summary','data/splits/split_summary_v2.json']
        for name, path in predictions.items():
            command += ['--predictions',name+'='+relative(self.root,path)]
        command += ['--output','{run_dir}/comparison.json','--report','{run_dir}/comparison.md','--csv','{run_dir}/comparison.csv']
        inputs = [relative(self.root,p) for p in predictions.values()]
        if test:
            inputs += ['data/processed/test.jsonl','docs/V2_FINAL_PROTOCOL.json']
        return self.job('holdout_evaluation' if test else 'dev_comparison',command,
                        ['comparison.json','comparison.md','comparison.csv'],stage='evaluate',
                        split='test' if test else 'validation',inputs=inputs,gpu=False,frozen=test)

    def freeze_and_holdout(self, calibrations, winner):
        ensure_clean(self.root)
        self.verify_source_snapshot()
        artifacts_by_family = {name:self.validate_decoder(name,directory/'calibration.json')
                               for name,directory in calibrations.items()}
        require(winner == select_family(artifacts_by_family), 'primary model differs from dev selection rule')
        models = {}
        for name, directory in calibrations.items():
            decoder = artifacts_by_family[name]
            models[name] = {'kind':name,'checkpoint':decoder['checkpoint'],
                            'calibration':relative(self.root,directory/'calibration.json')}
        artifacts = {p for model in models.values() for p in self.files(self.root/model['checkpoint'])}
        artifacts |= {m['calibration'] for m in models.values()}
        path = self.root/'docs/V2_FINAL_PROTOCOL.json'
        decision = {'schema_version':1,'status':'frozen','study_id':STUDY_ID,
            'cohort':'new internal holdout from original project train; NOT original official test',
            'seed':42,'split_seed':20261101,'models':models,'primary_selected_on_dev':winner,
            'frozen_utc':datetime.now(timezone.utc).isoformat(),
            'original_official_test_read':False,
            'artifacts_sha256':{p:sha256_file(self.root/p) for p in sorted(artifacts)},
            'inputs_sha256':dict(self.input_snapshot),
            'test_policy':'one prediction per fixed family and one common evaluation; no retuning from new holdout'}
        write_new(path,decision)
        commit_new_paths(self.root,[path],'Freeze V2 dev-selected models before new internal holdout')
        self.execute([str(self.cpu),'scripts/prepare_v2_data.py','--include-holdout'],'materialize_new_holdout')
        ensure_clean(self.root)
        self.verify_source_snapshot()
        for p,expected in decision['artifacts_sha256'].items():
            require(sha256_file(path_inside(self.root,p))==expected,'frozen V2 model artifact changed: '+p)
        predictions = {}
        for name, model in models.items():
            script = 'scripts/structured_transformer.py' if name=='structured' else 'scripts/transformer.py'
            command = ['{python}',script,'predict','--checkpoint',model['checkpoint'],'--calibration',model['calibration'],
                       '--split','test','--device','cuda','--processed-dir','data/processed',
                       '--manifest','data/splits/split_manifest_v2.csv','--split-summary','data/splits/split_summary_v2.json',
                       '--output','{run_dir}/predictions.jsonl']
            run = self.job('holdout_'+name,command,['predictions.jsonl'],split='test',
                inputs=[*artifacts,'data/processed/test.jsonl','docs/V2_FINAL_PROTOCOL.json'],frozen=True)
            predictions[name] = run/'predictions.jsonl'
        final = self.compare(predictions,test=True)
        self.state['holdout_comparison'] = relative(self.root,final/'comparison.json')

    def run(self):
        with original_queue_lock(self.original):
            self.run_locked()

    def run_locked(self):
        self.verify_source_snapshot()
        require(not (self.root/'data/processed/test.jsonl').exists(),'V2 holdout was already materialized')
        families = list(FAMILIES)
        for name in FAMILIES:
            try:
                _, pilot, _ = self.train(name,pilot=True)
                resources = pilot_resources(pilot)
                self.state.setdefault('pilot_resources',{})[name] = resources
                self.progress(name+'_pilot_resources',**resources)
                if name == 'structured':
                    require(resources['structured_cancellation_reason'] is None,
                            resources['structured_cancellation_reason'])
            except Exception as exc:
                if name != 'structured':
                    raise
                families.remove(name)
                self.state['canceled_families'][name] = 'train-only feasibility pilot failed: '+str(exc)
                self.progress('structured_canceled',reason=str(exc))
        calibrations = {}
        for name in families:
            _, _, checkpoints = self.train(name)
            calibrations[name] = self.calibrate(name,checkpoints)
        artifacts = {name:read_json(p/'calibration.json') for name,p in calibrations.items()}
        winner = select_family(artifacts)
        dev = self.compare({name:p/'validation_predictions.jsonl' for name,p in calibrations.items()})
        self.state.update(primary_selected_on_dev=winner,dev_comparison=relative(self.root,dev/'comparison.json'),
                          calibrations={name:relative(self.root,p) for name,p in calibrations.items()})
        self.freeze_and_holdout(calibrations,winner)
        self.verify_source_snapshot()
        snapshot = self.work/'delivery_state.json'
        write_new(snapshot,self.state)
        self.execute([str(self.cpu),'scripts/build_v2_delivery.py','--study',relative(self.root,snapshot)],
                     'assemble_v2_delivery')
        results = self.root/'results/v2'
        require(results.is_dir(), 'V2 delivery did not produce result reports')
        commit_new_paths(self.root,[path for path in sorted(results.rglob('*')) if path.is_file()],
                         'Publish V2 internal-holdout results and technical comparison delivery')
        self.verify_source_snapshot()
        self.state['visible_delivery'] = str(self.original/'artifacts/v2')
        self.state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(self.output,self.state)
        self.progress('completed',output=relative(self.root,self.output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/v2_study.json')
    args = parser.parse_args(argv)
    study = None
    try:
        study = Study(args.output)
        study.run()
    except (Exception,KeyboardInterrupt) as exc:
        if study is not None:
            study.state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            if not study.output.exists():
                write_new(study.output,study.state)
            write_state(study.status,study.state)
        parser.exit(1,f'V2 study stopped: {exc}\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
