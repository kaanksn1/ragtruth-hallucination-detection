# Geri bildirim sonrası tamamlayıcı validation benchmark'ı — 5 Kasım 2026

HMM, RNN, CNN ve ayrı klasik BERT, değerlendirme geri bildiriminden **sonra**
eklenen baseline'lardır. Kaynaklı/yalnız-cevap CRF ve ModernBERT için tarihsel
V1 validation tahminleri aynı karşılaştırmaya alınır. Eski resmî test sonuçları,
`docs/FINAL_PROTOCOL.json` ve özgün deneyin seçim kararları değiştirilmez.

**Tamamlandı:** HMM, RNN, CNN ve klasik BERT'in tam veri koşumları, BERT
validation kalibrasyonu ve yedi sabit tahminin ortak değerlendirmesi tamamlandı.
Bu dizin gerçek tahminleri, ölçülmüş sonuçları ve koşum kimliklerini içerir.

## Ortak veri ve metrikler

Bütün eğitimler özgün kaynak gruplu V1 ayrımını kullanır: `quality=good` olan
**12.702 eğitim ve 2.240 validation cevabı**. Dosyalardaki toplam sayılar
12.828 / 2.262'dir; kalite filtresi satırları ham dosyadan silmez. Bölüm ve
veri kimlikleri tarihsel çalışmayla aynıdır:

| Dosya | SHA-256 |
|---|---|
| `data/processed/train.jsonl` | `005def627fa45493e024e1b9e2295d289a0698ea475b888e8e2320d86151bb96` |
| `data/processed/validation.jsonl` | `d070d62056b2a7f11a58748291f6fd1881cceabc35e3db29435f1ae7d488c5ec` |
| `data/splits/split_manifest_v1.csv` | `7d292931c9337c9a4e30e9a62c7f09e72e8e8c51e31bc22b0522515aa3248a5f` |

Ortak scorer `ragtruth-project-eval-v1`, değiştirilmemiş cevap metnindeki
Unicode karakterlerini kullanır; boşluklar ve dört gerçek anotasyon türü
korunur. Accuracy, sorunsuz tokenların çoğunlukta olması nedeniyle ana ölçüt
değildir. Yeni RNN/CNN/BERT ve tarihsel ModernBERT seçiminde türlere eşit
ağırlık veren typed character macro F1 kullanılır. Tarihsel CRF ayar seçimi
any-character F1 kullanmıştı; HMM'nin ayarları sabittir. Bu seçim farkları ve
farklı arama bütçeleri korunur;
any-character F1 ve exact typed-span F1 farklı hedefleri ayrıca gösterir.
[Metrik gerekçesi](../../docs/METRIC_RATIONALE.md) ve
[tam değerlendirme sözleşmesi](../../docs/EVALUATION.md).

## Ölçülmüş validation sonuçları

Değerler **F1 yüzdesidir**, accuracy veya resmî test skoru değildir. Tarihsel
üç satır `results/validation_comparison.json` içindeki seçilmiş V1 tahminleridir;
ModernBERT için kalibre edilmiş V1 modeli kullanılır.

| Model | Girdi | Any-character F1 (%) | Typed macro F1 (%) | Exact typed-span F1 (%) |
|---|---|---:|---:|---:|
| HMM — yeni | Yalnız cevap | 21,02 | 10,94 | 0,52 |
| RNN — yeni | Yalnız cevap | 18,08 | 10,58 | 1,27 |
| CNN — yeni | Yalnız cevap | 18,99 | 11,48 | 1,42 |
| Klasik BERT — yeni | Kaynak + cevap | 41,16 | 22,62 | 2,03 |
| CRF — tarihsel V1 | Yalnız cevap | 23,10 | 13,13 | 5,28 |
| CRF — tarihsel V1 | Kaynak özellikleri + cevap | 39,08 | 19,99 | 6,41 |
| ModernBERT — tarihsel V1 | Kaynak + cevap | 54,05 | 32,01 | 7,00 |

HMM sabit `alpha=0.1`, en az iki kez görülen kelimelerden train vocabulary ve
dört ayrı BIO HMM/Viterbi kullanır; validation hiperparametre araması yoktur.
Tam fit/tahmin/değerlendirme **9,87 saniye** sürdü.

RNN ve CNN rastgele başlatılmış 128 boyutlu kelime embedding'leri, dört sigmoid
çıktı ve padding maskeli ağırlıksız BCE kullanır. Üç tam epoch, batch32,
Adam `1e-3`, seed42 ve 256 kelimelik eksiksiz parçalarla eğitildi. Vocabulary
yalnız train'den öğrenilir. Her epoch için global eşik
`[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]` validation macro F1, ardından any F1 ile
seçildi; eşitlikte önceki eşik/epoch korunur. Her ikisinde seçilen checkpoint
**epoch3, eşik0.1** oldu; normal yükleme yolundan yeniden tahminle skorlar
doğrulandı. RNN toplam **123,02 saniye**, CNN **59,28 saniye** sürdü; ortam
AMD Radeon RX 7800 XT, WSL2, Python3.12.3, Torch2.9.1+ROCm7.2.1'dir.

Klasik BERT, `google-bert/bert-base-uncased` revision
`86b5e0934494bd15c9632b12f734a8a67f723594` ile başlatılır. İlan edilmiş bütçe
bir tam epoch, FP32, batch8/accumulation1, AdamW `2e-5`, seed42 ve en fazla512
tokendır. Cevaplar kesilmeden pencerelenir; kaynak penceresi seçimi yalnız
model girdilerine dayanır. Ayrı validation kalibrasyonu tarihsel macro-first
kuralla yapıldı. Seçilen tür eşikleri EC/SC/EBI/SBI sırasıyla
**0.2 / 0.005 / 0.3 / 0.02**'dir. Tam eğitim geçişi **1.451,35 saniye**,
validation loss geçişi **317,70 saniye** sürdü. Eğitim/validation/kaydedilmiş
modelden varsayılan tahmin üretiminin toplamı **2.079,94 saniye**, ayrı
kalibrasyonun toplamı **322,07 saniye** oldu. Gerçek sayılar: train'de
12.706 pencere ve 1.589 optimizer adımı; validation'da 6.882 pencere.

Aynı BERT checkpoint'inde sabit 0.5 eşik any F1 **%23,41**, macro F1 **%12,80**
verirken validation ile seçilen eşikler **%41,16 / %22,62** verdi. Bunlar aynı
validation'da ayarlanmış sonuçlardır; bağımsız test kazanımı değildir. Varsayılan
tahminler ve skorları da `bert/default_predictions.jsonl` ve
`bert/default_comparison.json` ile saklanır.

## Kanıt dosyaları ve yeniden üretim

Bu dizindeki `comparison.json`, `comparison.md`
ve `comparison.csv` yedi modelin sabit tahminlerini ortak scorer ile karşılaştırır.
`predictions/{hmm,rnn,cnn,bert,crf_response_only,crf_source,modernbert_v1}.jsonl`
tam 2.240 uygun validation cevabını içerir. Model alt dizinleri gerçek koşum
özeti/konfigürasyon ve seçim kanıtlarını; `artifact_manifest.json` dosya
SHA-256 değerlerini ve özgün yerel tahmin yollarını saklar. Dosyaların tamamı
tamamlanmış koşumlardan kopyalandı; HMM/RNN/CNN rapor metrikleri yeniden
hesaplamayla birebir, BERT kalibrasyonunun karakter metrikleri de `1e-12`
toleransıyla eşleşti. Her modelde aynı 2.240 uygun validation ID'si bulunur.

[Ortak skor tablosu](comparison.md) · [JSON](comparison.json) ·
[CSV](comparison.csv) · [Karşılaştırma grafiği](figures/model_comparison.png) ·
[Tür grafiği](figures/type_f1.png) · [Görev grafiği](figures/task_f1.png).

`dataset_verification.json`, teslim edilen dokuz ham/işlenmiş veri dosyasının
byte/hash/satır sayısı ve Git'te izlenmesini; `git_data_audit.json`, Git veri
teslimi kontrolünü kayıt altına alır. Veri doğrulayıcısı test anotasyonlarını
parse etmez veya model performansı hesaplamaz:

```bash
python scripts/verify_submission_data.py --require-tracked
```

Depo kökünden aşağıdaki **Bash**
komutu yeniden skor üretir. Çıktı yeni bir `outputs/` dizinine yazılır;
commit edilmiş kanıtların üzerine yazılmaz. Aynı hedef zaten varsa farklı
bir çıktı dizini seçilmelidir.

```bash
python evaluate.py --split validation \
  --predictions hmm=results/supplement_2026-11-05/predictions/hmm.jsonl \
  --predictions rnn=results/supplement_2026-11-05/predictions/rnn.jsonl \
  --predictions cnn=results/supplement_2026-11-05/predictions/cnn.jsonl \
  --predictions bert=results/supplement_2026-11-05/predictions/bert.jsonl \
  --predictions crf_response_only=results/supplement_2026-11-05/predictions/crf_response_only.jsonl \
  --predictions crf_source=results/supplement_2026-11-05/predictions/crf_source.jsonl \
  --predictions modernbert_v1=results/supplement_2026-11-05/predictions/modernbert_v1.jsonl \
  --output outputs/supplement_replay_20261105/comparison.json \
  --report outputs/supplement_replay_20261105/comparison.md \
  --csv outputs/supplement_replay_20261105/comparison.csv
```

Yeni eğitim komutları ve ayrıntılı model tanımları:
[HMM](../../docs/BASELINE_HMM.md), [RNN/CNN](../../docs/BASELINE_NEURAL.md),
[klasik BERT](../../docs/BASELINE_BERT.md). Çalıştırma hedefleri yeni olmalıdır;
eğitilmiş ağırlıklar yerel `outputs/` altında kalır ve Git'e eklenmez.

## Kontroller ve yorum sınırı

Tamamlanan yerel test çalışmaları toplam **272 test** içerir: HF/CPU ortamında
**269 geçti, 3 atlandı — 28,544 saniye**; native/CRF ortamında
**234 geçti, 38 atlandı — 31,110 saniye**. Atlanan testler geçmiş sayılmaz;
iki ortamın sonuçları ayrı tutulur. İlk düzeltme commit'i `4302236` için dört GitHub
kontrolü geçti. Sonraki teslimin GitHub kontrolleri ilgili pull request'te
görülebilir; bu tarihsel kayıt ilk kod doğrulamasını sonradan değiştirmez.

HMM/RNN/CNN ve yalnız-cevap CRF kaynak okumaz; kaynaklı CRF/BERT/ModernBERT
ile fark hem girdi hem mimari farkını içerir. Böyle bir karşılaştırma tek
başına kaynak etkisi veya mimari üstünlük kanıtı değildir. Klasik BERT'in
bir epoch/512 token bütçesi, ModernBERT'in üç epoch/2.048 token bütçesiyle
eşit değildir; bütün modeller eşit ölçüde optimize edildiği iddia edilmez.

Yeni baseline'larda resmî test inference/değerlendirmesi yapılmaz. Validation,
eşik/checkpoint seçimi için kullanıldığı için bağımsız genelleme ölçümü
değildir. Ekler geri bildirim sonrasında ve eski test sonuçları biliniyorken
yapılmıştır; ilk teslimin önceden dondurulmuş deneyleri gibi sunulmaz.
