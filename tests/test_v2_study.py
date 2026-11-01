"""Offline guards for the V2 supervisor; no model or holdout is opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_v2_study as study


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value,str) else value, encoding='utf-8')


def artifact(value=.3, any_f1=.5):
    return {'schema_version':1,'status':'selected_on_validation','validation_split':'validation',
            'label_types':list(study.LABEL_TYPES),'selection_metric':'macro_f1_types',
            'validation_metrics':{'macro_f1_types':value,'any':{'f1':any_f1},
                                  **{name:{'f1':value} for name in study.LABEL_TYPES}}}


class SelectionTests(unittest.TestCase):
    def test_pilot_linear_budget_memory_and_finite_guards(self):
        summary={'epochs':[{'train':{'elapsed_seconds':32,'peak_allocated_bytes':8*1024**3},
                            'validation':{'elapsed_seconds':16,'peak_allocated_bytes':5*1024**3}}]}
        resources=study.pilot_resources(summary)
        self.assertEqual(resources['estimated_full_three_epoch_seconds'],3*(11431+2240))
        self.assertIn('18000',resources['structured_cancellation_reason'])
        summary['epochs'][0]['train']['elapsed_seconds']=1
        summary['epochs'][0]['validation']['elapsed_seconds']=1
        self.assertIsNone(study.pilot_resources(summary)['structured_cancellation_reason'])
        summary['epochs'][0]['train']['peak_allocated_bytes']=14*1024**3+1
        self.assertIn('14 GiB',study.pilot_resources(summary)['structured_cancellation_reason'])
        for invalid in (float('nan'),float('inf'),0,-1,True):
            summary['epochs'][0]['train']['elapsed_seconds']=invalid
            with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,'elapsed'):
                study.pilot_resources(summary)

    def test_finite_fixed_four_class_metrics_and_ties(self):
        rows = {name:artifact() for name in study.FAMILIES}
        self.assertEqual(study.select_family(rows),'bce')
        rows['focal'] = artifact(.31,.1)
        self.assertEqual(study.select_family(rows),'focal')
        rows['structured'] = artifact(.31,.2)
        self.assertEqual(study.select_family(rows),'structured')
        for invalid in (float('nan'),float('inf'),-.01,1.01,True):
            bad=artifact();bad['validation_metrics']['any']['f1']=invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                study.metric_key(bad)
        bad=artifact();bad['validation_metrics']['macro_f1_types']=.9
        with self.assertRaisesRegex(ValueError,'four classes'):
            study.metric_key(bad)
        for key,value in [('status','test_selected'),('validation_split','test'),('label_types',[])]:
            bad=artifact();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                study.metric_key(bad)

    def test_lock_is_exclusive_and_released_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'stop'):
                with study.original_queue_lock(root):
                    self.assertTrue((root/'outputs/.run.lock').is_file())
                    with self.assertRaisesRegex(ValueError,'another original'):
                        with study.original_queue_lock(root):
                            self.fail('second owner acquired lock')
                    raise RuntimeError('stop')
            self.assertFalse((root/'outputs/.run.lock').exists())

    def test_replaced_lock_is_not_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with study.original_queue_lock(root):
                write(root/'outputs/.run.lock',{'pid':-1,'scope':'other process'})
            self.assertEqual(study.read_json(root/'outputs/.run.lock')['pid'],-1)


class StudyFixture(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        original=Path(self.temporary.name)/'project'
        self.root=original/'.cache/ragtruth-v2'
        self.root.mkdir(parents=True)
        write(original/'.venv/Scripts/python.exe','fake interpreter never executed')
        self.tracked=['transformer.py','scripts/run_v2_study.py','configs/v2/bce.json',
                      'configs/v2/focal.json','configs/v2/structured.json','requirements.txt']
        for name in self.tracked+['AGENTS.md','docs/V2_PLAN.md',*study.DATA[:3]]:
            write(self.root/name,'fixture')
        write(self.root/study.DATA[3],{'study_id':study.STUDY_ID,'seed':20261101,
              'original_official_test_annotations_read':False,
              'processed_sha256':{side+'.jsonl':study.sha256_file(self.root/f'data/processed/{side}.jsonl')
                                  for side in ('train','validation')}})
        self.addCleanup(patch.stopall)
        patch.object(study,'ROOT',self.root).start()
        patch.object(study,'ensure_clean').start()
        patch.object(study,'git',side_effect=lambda *_:('\n'.join(self.tracked)+'\n').encode()).start()
        self.study=study.Study('outputs/study.json')

    def checkpoint(self, family='bce',epoch=1):
        directory=self.root/f'outputs/runs/{family}-train/checkpoints/epoch-{epoch}'
        write(directory/'head.safetensors','tiny fake weights')
        config={'revision':study.REVISION,'seed':42,'loss':family,'split_manifest':study.DATA[2],
                'split_summary':study.DATA[3],'processed_dir':'data/processed'}
        write(directory/'checkpoint.json',{'label_types':list(study.LABEL_TYPES),'config':config,
              'metadata':{'data_sha256':self.study.data_identity,'epoch':epoch},
              'files_sha256':{'head.safetensors':study.sha256_file(directory/'head.safetensors')}})
        return directory

    def calibration(self,family='bce'):
        checkpoint=self.checkpoint(family)
        decoder=artifact()
        decoder.update(checkpoint=study.relative(self.root,checkpoint),
                       checkpoint_sha256=study.sha256_file(checkpoint/'checkpoint.json'),
                       data_sha256=self.study.data_identity,threshold_grid=study.GRID,
                       thresholds_by_type={name:.5 for name in study.LABEL_TYPES})
        directory=self.root/f'outputs/runs/{family}-calibration'
        write(directory/'calibration.json',decoder)
        write(directory/'validation_predictions.jsonl','saved dev predictions')
        return directory

    def test_source_snapshot_rejects_edits_and_new_source(self):
        self.study.verify_source_snapshot()
        write(self.root/'transformer.py','changed')
        with self.assertRaisesRegex(ValueError,'study input changed'):
            self.study.verify_source_snapshot()
        write(self.root/'transformer.py','fixture')
        self.tracked.append('scripts/new.py')
        with self.assertRaisesRegex(ValueError,'input set changed'):
            self.study.verify_source_snapshot()

    def test_dynamic_final_jobs_do_not_change_scientific_inputs(self):
        self.tracked+=['configs/v2/final/holdout_bce.json','docs/V2_FINAL_PROTOCOL.json']
        self.study.verify_source_snapshot()
        self.assertNotIn('configs/v2/final/holdout_bce.json',self.study.inputs())

    def test_paths_and_existing_output_rejected(self):
        for name in ('../escape.json','outputs','outputs/run.txt','data/processed/a.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                study.Study(name)
        with self.assertRaisesRegex(ValueError,'existing study output'):
            study.Study('outputs/study.json')
        with self.assertRaisesRegex(ValueError,'missing or external'):
            self.study.files(self.root/'missing-cache')

    def test_decoder_checksum_and_data_guards(self):
        directory=self.calibration()
        path=directory/'calibration.json'
        good=study.read_json(path)
        self.assertEqual(self.study.validate_decoder('bce',path),good)
        for key,value in [('data_sha256',{}),('checkpoint_sha256','0'*64),
                          ('checkpoint','../outside'),('threshold_grid',[.5]),
                          ('thresholds_by_type',{name:True for name in study.LABEL_TYPES}),
                          ('official_test_evaluated',True)]:
            bad=copy.deepcopy(good);bad[key]=value;write(path,bad)
            with self.subTest(key=key),self.assertRaises((ValueError,FileNotFoundError)):
                self.study.validate_decoder('bce',path)
        write(path,good)
        write(self.root/good['checkpoint']/'head.safetensors','mutated')
        with self.assertRaisesRegex(ValueError,'file checksum'):
            self.study.validate_decoder('bce',path)

    def make_training(self,pilot=False):
        checkpoints=[self.checkpoint(epoch=i) for i in range(1,2 if pilot else 4)]
        run=checkpoints[0].parents[1]
        config=study.read_json(checkpoints[0]/'checkpoint.json')['config']
        summary={'status':'complete','config':config,'epochs':[{}]*(1 if pilot else 3),
                 'label_types':list(study.LABEL_TYPES),'data_sha256':self.study.data_identity,
                 'counts':{'train':32,'validation':16} if pilot else {'train':11431,'validation':2240},
                 'limits':{'train':32,'validation':16} if pilot else {'train':None,'validation':None},
                 'saved_checkpoints':[p.relative_to(run).as_posix() for p in checkpoints]}
        write(run/'transformer_summary.json',summary)
        snapshot=self.root/('.cache/hf-hub/models--answerdotai--ModernBERT-base/snapshots/'+study.REVISION)
        for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
            write(snapshot/name,'fake pretrained')
        patch.object(study,'PRETRAINED_SHA256',study.sha256_file(snapshot/'model.safetensors')).start()
        return run,summary

    def test_training_full_counts_limits_and_command(self):
        run,summary=self.make_training()
        with patch.object(self.study,'job',return_value=run) as job:
            self.study.train('bce')
            command=job.call_args.args[1]
            self.assertEqual(command[:5],['{python}','scripts/transformer.py','train','--config','configs/v2/bce.json'])
            self.assertNotIn('--limit-train',command)
            self.assertEqual(job.call_args.kwargs['split'],'train')
            self.assertEqual(job.call_args.kwargs['stage'],'train')
            self.assertEqual(len(job.call_args.kwargs['inputs']),5)
            for field,value in [('counts',{'train':11430,'validation':2240}),
                                ('limits',{'train':100,'validation':None}),('saved_checkpoints',[])]:
                bad=copy.deepcopy(summary);bad[field]=value;write(run/'transformer_summary.json',bad)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    self.study.train('bce')

    def test_pilot_is_bounded_and_wrong_pretrained_rejected(self):
        run,_=self.make_training(pilot=True)
        with patch.object(self.study,'job',return_value=run) as job:
            self.study.train('bce',pilot=True)
            self.assertEqual(job.call_args.args[1][-6:],['--limit-train','32','--limit-validation','16','--epochs','1'])
            self.assertEqual(job.call_args.kwargs['timeout'],900)
            with patch.object(study,'PRETRAINED_SHA256','wrong'):
                with self.assertRaisesRegex(ValueError,'original pretrained'):
                    self.study.train('bce')

    def test_job_record_bound_to_initial_inputs_and_native_argv(self):
        run=self.root/'outputs/runs/fake';run.mkdir(parents=True)
        write(run/'comparison.json',{})
        def execute(argv,name):
            self.study.progress(name)
            self.assertEqual(argv[:2],[str(self.study.cpu),'scripts/run_job.py'])
            config=study.read_json(self.root/argv[2])
            write(run/'run.json',{'status':'completed','exit_code':0,'config':config,
                  'inputs_sha256':{p:study.sha256_file(self.root/p) for p in config['inputs']},
                  'outputs_sha256':{'comparison.json':study.sha256_file(run/'comparison.json')}})
            return json.dumps({'status':'completed','exit_code':0,'record':str(run/'run.json')})
        with patch.object(self.study,'execute',side_effect=execute):
            self.assertEqual(self.study.job('dev',['{python}','evaluate.py'],['comparison.json'],gpu=False),run)
        with self.assertRaisesRegex(ValueError,'frozen protocol'):
            self.study.job('bad',[],[],split='test')

    def test_wsl_job_uses_argv_and_pinned_offline_environment(self):
        run=self.root/'outputs/runs/fake';run.mkdir(parents=True)
        write(run/'result.json',{})
        def execute(argv,name):
            self.study.progress(name)
            self.assertEqual(argv[:7],['wsl.exe','--distribution','Ubuntu-24.04','--user','serda','--cd',
                                      study.windows_to_wsl(str(self.root))])
            self.assertEqual(argv[7:9],['--','env'])
            self.assertIn('HF_HUB_OFFLINE=1',argv)
            self.assertIn('TRANSFORMERS_OFFLINE=1',argv)
            self.assertEqual(argv[-3:-1],['/opt/banking77-venv/bin/python','scripts/run_job.py'])
            config=study.read_json(self.root/argv[-1])
            write(run/'run.json',{'status':'completed','exit_code':0,'config':config,
                  'inputs_sha256':{p:study.sha256_file(self.root/p) for p in config['inputs']},
                  'outputs_sha256':{'result.json':study.sha256_file(run/'result.json')}})
            return json.dumps({'status':'completed','exit_code':0,'record':str(run/'run.json')})
        # On Linux the fake-root path is POSIX; exercise the same argv with a
        # stand-in converter, while the actual pure Windows conversion is checked below.
        with (patch.object(self.study,'execute',side_effect=execute),
              patch.object(study,'windows_to_wsl',return_value='/mnt/c/fixture with spaces/ç')):
            self.study.job('gpu_fixture',['{python}','train.py'],['result.json'])
        self.assertEqual(study.windows_to_wsl(r'C:\Users\with spaces\ç\repo'),'/mnt/c/Users/with spaces/ç/repo')

    def test_structured_calibration_grid_and_validation_only_command(self):
        checkpoint=self.checkpoint('structured')
        decoder=artifact()
        decoder.update(checkpoint=study.relative(self.root,checkpoint),
                       checkpoint_sha256=study.sha256_file(checkpoint/'checkpoint.json'),
                       data_sha256=self.study.data_identity,o_bias_grid=study.BIAS_GRID,
                       o_bias_by_type={name:0 for name in study.LABEL_TYPES})
        run=self.root/'outputs/runs/structured-calibration'
        write(run/'calibration.json',decoder)
        with patch.object(self.study,'job',return_value=run) as job:
            self.study.calibrate('structured',[checkpoint])
            command=job.call_args.args[1]
            self.assertEqual(command[:3],['{python}','scripts/structured_transformer.py','calibrate'])
            config=study.read_json(self.root/command[command.index('--config')+1])
            self.assertEqual(config,{'split':'validation','checkpoints':[study.relative(self.root,checkpoint)],
                                     'o_bias_grid':[-8,-6,-4,-2,0,2,4,6,8]})
        decoder['o_bias_grid']=[-2,-1,0,1,2]
        write(run/'calibration.json',decoder)
        with self.assertRaisesRegex(ValueError,'structured decoder grid'):
            self.study.validate_decoder('structured',run/'calibration.json')

    def test_compare_and_freeze_use_v2_holdout_only_after_commit(self):
        calibrations={name:self.calibration(name) for name in ('bce','focal')}
        events=[];calls=[]
        def execute(command,name):
            self.assertEqual(events,['commit'])
            decision=study.read_json(self.root/'docs/V2_FINAL_PROTOCOL.json')
            self.assertEqual(decision['study_id'],study.STUDY_ID)
            self.assertEqual(decision['inputs_sha256'],self.study.input_snapshot)
            self.assertFalse(decision['original_official_test_read'])
            self.assertEqual(command[1:],['scripts/prepare_v2_data.py','--include-holdout'])
            events.append('prepare')
        def job(name,command,outputs,**kwargs):
            self.assertEqual(events,['commit','prepare'])
            calls.append((name,command,kwargs))
            self.assertEqual(kwargs['split'],'test')
            self.assertTrue(kwargs['frozen'])
            self.assertIn('data/processed/test.jsonl',kwargs['inputs'])
            self.assertIn('data/splits/split_manifest_v2.csv',command)
            self.assertIn('data/splits/split_summary_v2.json',command)
            return self.root/'outputs/runs'/name
        with (patch.object(study,'commit_new_paths',side_effect=lambda *_:events.append('commit')),
              patch.object(self.study,'execute',side_effect=execute),patch.object(self.study,'job',side_effect=job)):
            self.study.freeze_and_holdout(calibrations,'bce')
        self.assertEqual([name for name,_,_ in calls],['holdout_bce','holdout_focal','holdout_evaluation'])
        self.assertEqual(calls[-1][2]['stage'],'evaluate')
        self.assertFalse(calls[-1][2]['gpu'])

    def test_wrong_winner_rejected_before_holdout(self):
        calibrations={name:self.calibration(name) for name in ('bce','focal')}
        with patch.object(self.study,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'dev selection'):
                self.study.freeze_and_holdout(calibrations,'focal')
            execute.assert_not_called()
            self.assertFalse((self.root/'docs/V2_FINAL_PROTOCOL.json').exists())


if __name__=='__main__':
    unittest.main()
