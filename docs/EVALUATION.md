# Ortak proje değerlendirmesi

Sürüm: `ragtruth-project-eval-v1`. Kod `ragtruth_eval/`, komutlar `evaluate.py`
ve `plot_results.py` içindedir. Scorer model eğitmez ve eşik seçmez; sabit
tahmin dosyalarını aynı kayıtlar ve metriklerle karşılaştırır.

Bu proje, kaynak içeriğe göre cevaptaki anotasyonlu aralıkları bulur. Kaynakta
bulunmayan fakat dünyada doğru olabilen `implicit_true` anotasyonları ve
`due_to_null` işaretleri korunur. Dört gerçek tür ham adlarıyla değerlendirilir;
BIO etiket sayısı anlamsal sınıf sayısı olarak sunulmaz.

## Metrik tanımları

Karakter koordinatları, değiştirilmemiş orijinal cevapta Python Unicode code
point indeksleridir; UTF-8 byte sayısı veya tokenizer indeksi değildir.
Karakter aralıkları `[start,end)` biçimindedir. **Boşluklar dahildir**; scorer
strip/lowercase veya boşluk normalizasyonu yapmaz.

- **Character any micro P/R/F1:** Cevap başına tüm anotasyon türlerinin
  kapsadığı karakterlerin birleşimi alınır. Örtüşen karakterler bir kez
  sayılır. Cevapların TP/FP/FN sayıları toplanır; cevap F1'lerinin ortalaması
  alınmaz. Türü yanlış olsa da doğru konumdaki tahmin bu ikili metrikte doğrudur.
- **Tür bazında character P/R/F1:** Aynı işlem her tür için ayrı yapılır;
  farklı türler aynı karakterde birlikte doğru olabilir. Her türün gold ve
  predicted karakter desteği ile pozitif cevap sayısı yazılır.
- **Typed character macro F1:** Sabit dört türün F1 ortalamasıdır. Bir tür
  hem gold hem tahminde yoksa F1=0'dır ve yine ortalamaya katılır.
- **Typed character micro P/R/F1:** Dört türün TP/FP/FN sayıları toplanır;
  aynı karakter farklı türlerde ayrı değerlendirme birimleri olabilir.
- **Exact typed span P/R/F1:** Başlangıç, bitiş ve türün üçü de doğru olmalı.
  Cevap içindeki birebir aynı `(start,end,type)` tekrarları bir kez sayılır.
  Ham anotasyonların örtüşen veya komşu sınırları birleştirilmez. Örneğin
  gold `[0,3)` ve `[2,5)` iken tahmin `[0,5)` character F1=1 verebilir,
  fakat exact span eşleşmesi değildir.
- **Exact any span P/R/F1:** Tür dikkate alınmaz, birebir aynı `(start,end)`
  aralıkları tek sayılır. Ham sınırlar yine birleştirilmez.

Her metrikte `precision=TP/(TP+FP)`, `recall=TP/(TP+FN)` ve
`F1=2TP/(2TP+FP+FN)` kullanılır. Payda sıfırsa sonuç **0**'dır; boş gold ve
boş tahmin yapay olarak F1=1 üretmez. Tam eşleşme tekrarlarının deduplikasyonu
yalnız skor hesabındadır; ham dosya değiştirilmez. JSON ham ve benzersiz span
sayılarını ayrı saklar. `task_type` ve `generator` kırılımları aynı sabit
tahminlerin alt kümeleriyle üretilir; ayrı model veya eşik seçilmez.

Bu karakter metriği mevcut CRF ayar seçimi yardımcısıyla eşdeğerdir. Rastgele
örtüşen aralıklarda bağımsız interval hesabı ile eski karakter-set hesabının
paritesi test edilir. İlk Transformer dosyasında any character F1 yaklaşık
0,4464'tür; bu accuracy değildir.

## Makale ve upstream baseline ile ilişki

Bunlar açıkça tanımlanmış **proje metrikleridir**; makalenin scorer'ıyla veya
yayınlanmış tüm word-level sonuçlarıyla birebir eşdeğer oldukları iddia edilmez.
Pinned upstream [baseline/predict_and_evaluate.py](https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/baseline/predict_and_evaluate.py)
betiği cevap düzeyinde hallucination var/yok P/R/F1 hesaplar; bu, bizim
karakter konumu ve dört türü ayrı değerlendirmemizden farklıdır.
[baseline/prepare_dataset.py](https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/baseline/prepare_dataset.py)
`quality=good` filtrelemesi yapar ve etiket adlarını conflict/baseless info
biçimine indirger; bizim dört ham tür korunur. Bu dosyalardaki gözlemler,
makalenin tüm değerlendirmesinin yalnız bu betiklerden oluştuğu anlamına gelmez.

## Bütünlük ve veri politikası

CLI yalnız `quality=good` kayıtları değerlendirir. Filtre öncesi/sonrası
sayılar ve dışlanan kalite grupları JSON'a yazılır. Manifestin tamamında
aynı içerik grubunun bölümler arasında bulunması, yinelenen source_id,
kayıt/split uyuşmazlığı ve summary SHA-256 farkı reddedilir.

Gold ve tahmin response ID kümeleri birebir aynı olmalıdır. Eksik, fazla,
yinelenen veya string olmayan ID'ler reddedilir. Offset'ler bool/float olamaz;
negatif, sıfır uzunluklu veya cevap sınırını aşan aralıklar ve bilinmeyen türler
reddedilir. Gold `text` varsa orijinal offset dilimiyle aynı olmalıdır.
Mevcut çıktıların üzerine sessizce yazılmaz. Her modelin tahmin checksum'ı,
gold/manifest/summary ve scorer kod checksum'ları raporda tutulur.

## Validation komutu ve grafikler

Depo kökünde, mevcut JSONL dosyalarıyla:

```powershell
python evaluate.py --split validation --predictions crf=outputs/crf/validation_predictions.jsonl --predictions transformer=outputs/transformer/validation_predictions.jsonl --output outputs/comparison.json --report outputs/comparison.md --csv outputs/comparison.csv
python plot_results.py --input outputs/comparison.json --output-dir outputs/comparison_figures
```

Örnek yollar gerçek koşum dizinleriyle değiştirilir. `--gold`, `--manifest`
ve `--split-summary` ile özel fixture dosyaları verilebilir. Varsayılanlar
`data/processed/validation.jsonl`, `data/splits/split_manifest_v1.csv` ve
`data/splits/split_summary_v1.json`'dır. Evaluator standart kütüphane kullanır;
grafik aracı mevcut matplotlib/numpy ortamında çalışır.

Grafikler `model_comparison`, `type_f1`, `task_f1` adlarıyla PNG ve SVG olarak
üretilir. Eksik model sonucu sıfırmış gibi eklenmez; yalnız verilen modeller
karşılaştırılır. Örtüşen kayıtlar tüm modeller için aynı gold kümesinde bulunur.

## Resmî test kapısı

`--split test`, gold veya tahminleri parse etmeden önce `run_job` üzerinden
çalışan tam `evaluate` komutunu, temiz/commit'li kodu ve dondurulmuş protokolü
doğrular. Job girdilerinde evaluator paketinin tüm dosyaları, ortak veri
şeması/reader, runner, test gold, manifest, summary ve verilen her tahmin
dosyasının SHA-256 değeri zorunludur. Hazır olmayan veya eksik beyanlı test
komutu dosyaları açmadan durur. Teknik gate bir güvenlik sandbox'ı değildir;
protokolün bilimsel uygunluğu ayrıca inceleme gerektirir.

Orijinal `split_summary_v1.json`, bilerek yalnız train/validation dosyalarının
hash'lerini içerir ve checkpoint metadata'sı bu summary'nin hash'ine bağlıdır.
Final test için bu dosya değiştirilmez. Test dosyasının hash'i tam donmuş
`evaluate` job'ı tarafından doğrulandıktan sonra CLI reader'a
`frozen_test_hash_verified=True` verir. Bu durumda summary'de test hash'inin
bulunmaması kabul edilir; varsa yanlış hash yine reddedilir. Train/validation
hash zorunluluğu gevşetilmez. Rapor bu durumu
`gold_identity_source=frozen_test_job` olarak kaydeder.

Resmî test sonrasında model, eşik veya özellikler sonuçlara bakarak ayarlanmaz.
API'nin genel dosya okuyucuları kendi başlarına test izni vermez; final
değerlendirme, merkezi CLI/protokol yolundan yapılmalıdır.

## Python API

```python
from ragtruth_eval import evaluate_records, character_scores

metrics = evaluate_records(original_records, prediction_rows_or_id_to_spans)
any_f1 = metrics["character"]["any"]["f1"]
macro_f1 = metrics["character"]["macro_f1_types"]
exact_f1 = metrics["span_exact"]["typed"]["f1"]
```

`evaluate_records` zaten seçilmiş kayıtları değerlendirir; kendi başına quality
filtrelemez. Daha düşük seviyedeki `character_scores(gold_by_id,pred_by_id)`
ID/tür/offset sırasını doğrular; cevap metni olmadığı için cevap uzunluğu
kontrolü üstteki API/CLI'dedir. API sonuçları oran olarak 0–1 aralığındadır.
