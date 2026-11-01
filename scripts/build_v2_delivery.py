"""Package fixed V2 aggregates after the new internal holdout is scored."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ragtruth_data.data import read_jsonl,sha256_file
from ragtruth_eval.io import load_gold,read_predictions
from scripts.error_analysis import support,select_examples,markdown
from scripts.package_handoff import build_package
from ragtruth_eval import METRIC_VERSION

STUDY_ID='ragtruth-v2-fresh-original-train-holdout'
FAMILIES={'bce','focal','structured'}


def contained(root,value):
    path=(root/Path(value)).resolve()
    if path==root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError('delivery input must stay inside the V2 worktree')
    return path


def validate_comparisons(state,dev,holdout):
    """Validate aggregate identities without reading any holdout example/gold."""
    names=set(state.get('calibrations',{}))
    if not {'bce','focal'} <= names <= FAMILIES:
        raise ValueError('study has missing or unknown model families')
    if state.get('original_official_test_read') is not False:
        raise ValueError('original official test must remain excluded')
    if state.get('primary_selected_on_dev') not in names:
        raise ValueError('dev-selected primary is not a completed study family')
    if state.get('status') not in {'running','completed'}:
        raise ValueError('study has not completed its fixed comparison stages')
    if dev.get('split')!='validation' or holdout.get('split')!='test':
        raise ValueError('comparison split differs from the fixed study')
    identities=state.get('initial_inputs_sha256',{})
    expected={'split_manifest':identities.get('data/splits/split_manifest_v2.csv'),
              'split_summary':identities.get('data/splits/split_summary_v2.json')}
    if not all(isinstance(value,str) and value for value in expected.values()):
        raise ValueError('study has no frozen V2 data identity')
    for comparison in (dev,holdout):
        if set(comparison.get('models',{}))!=names:
            raise ValueError('comparison model families differ from the fixed study')
        if comparison.get('metric_version')!=METRIC_VERSION:
            raise ValueError('comparison metric version differs from current scorer')
        data=comparison.get('data',{})
        if data.get('quality_filter')!='good' or any(data.get('sha256',{}).get(k)!=v for k,v in expected.items()):
            raise ValueError('comparison cohort differs from frozen V2 data')
        if any(row.get('metrics',{}).get('responses')!=data.get('included_responses')
               for row in comparison['models'].values()):
            raise ValueError('model response cohort differs from comparison data')
    if dev['data']['sha256'].get('gold')!=identities.get('data/processed/validation.jsonl'):
        raise ValueError('dev comparison does not match frozen V2 validation bytes')
    if not holdout['data']['sha256'].get('gold') or holdout['data']['sha256']['gold']==dev['data']['sha256']['gold']:
        raise ValueError('holdout comparison must identify a distinct held-out cohort')


def metric_table(comparison):
    lines=['| Model | Any character F1 (%) | Four-type macro F1 (%) | Exact typed-span F1 (%) |',
           '|---|---:|---:|---:|']
    for name,row in comparison['models'].items():
        metrics=row['metrics'];c=metrics['character']
        lines.append(f"| {name} | {100*c['any']['f1']:.2f} | {100*c['macro_f1_types']:.2f} | {100*metrics['span_exact']['typed']['f1']:.2f} |")
    return lines


def report(state,dev,holdout):
    validate_comparisons(state,dev,holdout)
    chosen=state['primary_selected_on_dev']
    improvement=100*(holdout['models'][chosen]['metrics']['character']['macro_f1_types']-
                     holdout['models']['bce']['metrics']['character']['macro_f1_types'])
    return '\n'.join(['# V2: yeni geliştirme turunun sonuçları','',
        '**Bu, RAGTruth resmî testinin tekrar ölçümü değildir.** Yeni kontrol kümesi, önceki projenin '
        'train bölümünden kaynak gruplarıyla ayrılmıştır. Bütün V2 modeller bu gruplar çıkarıldıktan '
        'sonra ilk pretrained ağırlıklardan yeniden eğitildi. Eski resmî sonuç %52,06 olarak arşivde kalır; '
        'aşağıdaki farklı kümenin skorları doğrudan bu değerle kıyaslanmaz.','',
        f"Yeni holdout: {holdout['data']['included_responses']} good cevap / {holdout['data']['file_responses']} toplam. "
        f"Hariç tutulan kalite etiketleri: {holdout['data']['excluded_by_quality']}.",'',
        '## Dev — model ve decoder seçiminde kullanıldı','',*metric_table(dev),'',
        '## Yeni internal holdout — seçimler dondurulduktan sonra tek ölçüm','',*metric_table(holdout),'',
        f"Dev ile seçilen birincil model: **{chosen}**. Bu modelin yeni holdout dört-tür macro F1 farkı, "
        f"aynı yeni train/holdout ayrımındaki fresh BCE kontrolüne göre **{improvement:+.2f} yüzde puan**. "
        'Holdout tablosu sonradan başka bir modeli seçmek veya tekrar ayar yapmak için kullanılmadı.','',
        '## Ne değişti?','',
        '- BCE: dört bağımsız sigmoid çıkışlı kontrol.','- Focal: aynı mimari ve veri; gamma=2, alpha=0.75. '
        'Kolay hedeflerin loss katkısını azaltma hipotezi sınandı.',
        *(['- Structured: aynı fresh encoder, orijinal kelime aralıklarına mean pooling, dört bağımsız BIO CRF. '
           'Türler arası örtüşmeler korunur. Orijinal metin veya anotasyonlar değiştirilmez.']
          if 'structured' in dev['models'] else []),'',
        'Tablolardaki tamamlanmış her family üç epoch eğitildi ve bütün checkpointler saklandı. Model/decoder seçimi yalnız dev '
        'dört-tür character macro F1, ardından any F1, ardından erken epoch kuralıyla yapıldı. '
        'Focal/BCE için aynı 15 eşik kullanıldı. '
        +('Structured için sabit dokuz O-bias değeri kullanıldı. ' if 'structured' in dev['models'] else '')+
        'Daha iyi bir sonuç varsayılmadı; negatif denemeler de tabloda tutuldu.','',
        f"İptal edilen family/pilotlar: {state.get('canceled_families',{})}.",'',
        '## Sınırlar','',
        'Dört türün tek tek skorları [holdout tablosunda](holdout_comparison.md). Any karakter, tür macro '
        've tam span ölçümleri aynı şey değildir. Nadir türlerde sıfır F1 kalabilir; modelin her türü '
        'güvenilir çözdüğü iddia edilmez. Tek seed ve bu sınırlı deney bütçesi çoklu-seed güven aralığı '
        'sağlamaz. Farklı decoder aramaları ve model mimarileri birlikte değiştiğinde kazanımın tamamı '
        'tek bir parçaya atfedilemez.','',
        '## Dosyalar','',
        '[Dev karşılaştırması](dev_comparison.md), [holdout karşılaştırması](holdout_comparison.md), '
        '[dev hata örnekleri](error_analysis.md), [grafikler](figures/holdout/). '
        'Demo kaydedilmiş dev tahminlerini gösterir; canlı inference değildir. '
        'Kaynaklar ve önceden ilan edilmiş kurallar [V2_PLAN.md](../../docs/V2_PLAN.md) içindedir.','',
        'Deneyin bilimsel kararı `docs/V2_FINAL_PROTOCOL.json`, byte-bound job çiftleri '
        '`configs/v2/final/` altındadır. Orijinal resmî test burada açılmadı veya yeniden skorlanmadı.',''])


def build(study_path, *, root=ROOT):
    root=Path(root).resolve()
    study_path=contained(root,study_path)
    state=json.loads(study_path.read_text(encoding='utf-8'))
    require_keys=('dev_comparison','holdout_comparison','calibrations','primary_selected_on_dev')
    if any(key not in state for key in require_keys):
        raise ValueError('study must have completed both frozen comparison stages')
    if state.get('original_official_test_read') is not False:
        raise ValueError('original official test must remain excluded')
    target=root/'results/v2'
    original=root.parent.parent
    demo=root/'artifacts/v2-comparison-demo.html'
    bundle=root/'artifacts/ragtruth_v2_handoff.zip'
    visible=original/'artifacts/v2'
    if any(path.exists() for path in (target,demo,bundle,visible)):
        raise FileExistsError('refusing to overwrite an existing V2 delivery')
    dev_path,holdout_path=[contained(root,state[name]) for name in ('dev_comparison','holdout_comparison')]
    dev=json.loads(dev_path.read_text(encoding='utf-8'));holdout=json.loads(holdout_path.read_text(encoding='utf-8'))
    validate_comparisons(state,dev,holdout)
    manifest=root/'data/splits/split_manifest_v2.csv';summary=root/'data/splits/split_summary_v2.json'
    if json.loads(summary.read_text(encoding='utf-8')).get('study_id')!=STUDY_ID:
        raise ValueError('delivery must use the fresh V2 data cohort')
    for path,key in ((manifest,'split_manifest'),(summary,'split_summary')):
        if sha256_file(path)!=dev['data']['sha256'][key]:
            raise ValueError('current V2 manifest/summary differs from scored cohort')
    for side,comparison in (('validation',dev),('test',holdout)):
        # Hashes only: no holdout labels/examples are parsed by the builder.
        if sha256_file(root/f'data/processed/{side}.jsonl')!=comparison['data']['sha256']['gold']:
            raise ValueError('current data bytes differ from scored V2 cohort')
        for row in comparison['models'].values():
            if sha256_file(contained(root,row['predictions_path']))!=row['predictions_sha256']:
                raise ValueError('comparison prediction bytes changed')
    target.mkdir(parents=True,exist_ok=False)
    plot_python=original.parent/'.venv/Scripts/python.exe'
    cpu_python=original/'.venv/Scripts/python.exe'
    def run(argv):
        subprocess.run([str(p) for p in argv],cwd=root,check=True)
    for label,path in (('dev',dev_path),('holdout',holdout_path)):
        for suffix in ('json','md','csv'):
            shutil.copyfile(path.with_suffix('.'+suffix),target/f'{label}_comparison.{suffix}')
        run([plot_python,'plot_results.py','--input',path,'--output-dir',target/'figures'/label])
    loaded={side:load_gold(root/f'data/processed/{side}.jsonl',side,manifest=manifest,summary=summary)
            for side in ('train','validation')}
    data_summary=json.loads(summary.read_text(encoding='utf-8'))
    for side in ('train','validation'):
        name=f'crf_{side}.jsonl'
        if sha256_file(root/'data/processed'/name)!=data_summary['processed_sha256'].get(name):
            raise ValueError('word-view support data differs from frozen V2 summary')
    errors={'split':'validation','comparison_sha256':sha256_file(dev_path),
        'data_sha256':{side:value[1]['sha256'] for side,value in loaded.items()},
        'support':{side:support(value[0],read_jsonl(root/f'data/processed/crf_{side}.jsonl')) for side,value in loaded.items()},
        'models':{name:select_examples(loaded['validation'][0],read_predictions(contained(root,row['predictions_path'])))
                  for name,row in dev['models'].items()}}
    (target/'error_analysis.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    (target/'error_analysis.md').write_text(markdown(errors),encoding='utf-8',newline='\n')
    (target/'FINAL_REPORT.md').write_text(report(state,dev,holdout),encoding='utf-8',newline='\n')
    shutil.copyfile(study_path,target/'study.json')
    shutil.copyfile(manifest,target/manifest.name)
    shutil.copyfile(summary,target/summary.name)
    for family,directory in state['calibrations'].items():
        shutil.copyfile(contained(root,directory)/'calibration_report.json',target/f'{family}_calibration.json')
    command=[cpu_python,'scripts/build_demo.py','--records','data/processed/validation.jsonl']
    for family,row in dev['models'].items():
        command+=['--predictions',family+'='+row['predictions_path']]
    command+=['--output',demo]
    run(command)
    packaged=build_package(root,demo,target,bundle)
    visible.mkdir(parents=True,exist_ok=False)
    for source,name in ((demo,'demo.html'),(bundle,'ragtruth_v2_handoff.zip'),(target/'FINAL_REPORT.md','FINAL_REPORT.md')):
        shutil.copyfile(source,visible/name)
    return {'results':target.relative_to(root).as_posix(),'package':packaged,'visible_delivery':str(visible)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(build(args.study)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
