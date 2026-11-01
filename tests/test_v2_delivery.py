"""V2 aggregate delivery guards; no real gold, holdout or model files are read."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ragtruth_eval import METRIC_VERSION
from scripts.build_v2_delivery import build,report,validate_comparisons


def fixture(structured=True,chosen='bce'):
    families=['bce','focal']+(['structured'] if structured else [])
    state={'status':'completed','original_official_test_read':False,
        'primary_selected_on_dev':chosen,'calibrations':{name:'fake/'+name for name in families},
        'initial_inputs_sha256':{'data/splits/split_manifest_v2.csv':'manifest-v2',
            'data/splits/split_summary_v2.json':'summary-v2','data/processed/validation.jsonl':'dev-gold'},
        'dev_comparison':'fake/dev.json','holdout_comparison':'fake/holdout.json'}
    def comparison(split):
        return {'split':split,'metric_version':METRIC_VERSION,'data':{'quality_filter':'good',
            'included_responses':20,'file_responses':22,'excluded_by_quality':{'truncated':2},
            'sha256':{'split_manifest':'manifest-v2','split_summary':'summary-v2',
                      'gold':'dev-gold' if split=='validation' else 'fresh-holdout'}},
            'models':{name:{'metrics':{'responses':20,'character':{
                'any':{'f1':.6},'macro_f1_types':.4 if name=='bce' else .5},
                'span_exact':{'typed':{'f1':.1}}}} for name in families}}
    return state,comparison('validation'),comparison('test')


class V2DeliveryTests(unittest.TestCase):
    def test_primary_uses_dev_state_even_if_holdout_prefers_another_model(self):
        state,dev,holdout=fixture(chosen='bce')
        holdout['models']['structured']['metrics']['character']['macro_f1_types']=.99
        output=report(state,dev,holdout)
        self.assertIn('birincil model: **bce**',output)
        self.assertIn('**+0.00 yüzde puan**',output)
        self.assertIn('sabit dokuz O-bias',output)
        self.assertEqual(state['primary_selected_on_dev'],'bce')

    def test_canceled_structured_is_not_reported_as_completed(self):
        state,dev,holdout=fixture(structured=False,chosen='focal')
        state['canceled_families']={'structured':'train-only feasibility pilot failed'}
        output=report(state,dev,holdout)
        self.assertNotIn('- Structured:',output)
        self.assertNotIn('O-bias değeri kullanıldı',output)
        self.assertIn('train-only feasibility pilot failed',output)
        self.assertIn('**+10.00 yüzde puan**',output)

    def test_model_family_mismatches_fail(self):
        for mutation in ('missing','unknown','primary','state'):
            with self.subTest(mutation=mutation):
                state,dev,holdout=fixture()
                if mutation=='missing':
                    del holdout['models']['structured']
                elif mutation=='unknown':
                    state['calibrations']['old_transformer']='fake/old'
                elif mutation=='primary':
                    state['primary_selected_on_dev']='best-from-test'
                else:
                    state['status']='failed'
                with self.assertRaises(ValueError):
                    report(state,dev,holdout)

    def test_cohort_mismatches_fail(self):
        for mutation in ('split','manifest','summary','gold','same_gold','responses','metric','quality','old_test'):
            with self.subTest(mutation=mutation):
                state,dev,holdout=fixture()
                if mutation=='split':
                    dev['split']='test'
                elif mutation in ('manifest','summary'):
                    holdout['data']['sha256']['split_'+mutation]='v1'
                elif mutation=='gold':
                    dev['data']['sha256']['gold']='different-dev'
                elif mutation=='same_gold':
                    holdout['data']['sha256']['gold']='dev-gold'
                elif mutation=='responses':
                    holdout['models']['bce']['metrics']['responses']=19
                elif mutation=='metric':
                    holdout['metric_version']='other-metric'
                elif mutation=='quality':
                    holdout['data']['quality_filter']='all'
                else:
                    state['original_official_test_read']=True
                with self.assertRaises(ValueError):
                    validate_comparisons(state,dev,holdout)

    def test_existing_delivery_fails_before_gold_or_subprocess_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'original/.cache/ragtruth-v2'
            root.mkdir(parents=True)
            state,_,_=fixture()
            study=root/'outputs/study.json'
            study.parent.mkdir()
            study.write_text(json.dumps(state),encoding='utf-8')
            for relative in ('results/v2','artifacts/v2-comparison-demo.html',
                             'artifacts/ragtruth_v2_handoff.zip','../../artifacts/v2'):
                with self.subTest(relative=relative):
                    target=root/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(b'preserved')
                    with patch('scripts.build_v2_delivery.load_gold') as gold, \
                         patch('scripts.build_v2_delivery.subprocess.run') as process:
                        with self.assertRaisesRegex(FileExistsError,'overwrite'):
                            build(study,root=root)
                        gold.assert_not_called()
                        process.assert_not_called()
                    self.assertEqual(target.read_bytes(),b'preserved')
                    target.unlink()

    def test_invalid_comparison_does_not_create_delivery_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'original/.cache/ragtruth-v2'
            root.mkdir(parents=True)
            state,dev,holdout=fixture()
            holdout['data']['sha256']['split_summary']='old-v1'
            for relative,value in (('outputs/study.json',state),('fake/dev.json',dev),('fake/holdout.json',holdout)):
                path=root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'cohort'):
                build(root/'outputs/study.json',root=root)
            self.assertFalse((root/'results/v2').exists())


if __name__=='__main__':
    unittest.main()
