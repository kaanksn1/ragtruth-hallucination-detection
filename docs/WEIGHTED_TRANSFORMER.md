# Tek, önceden tanımlanmış ağırlıklı BCE deneyi

Amaç, ilk modelin az görülen türlerdeki düşük recall'unu inceleyen tek kontrollü
deneydir. Sonucun iyileşeceği garanti edilmez. Anotasyonlar, örnek sayısı,
split, encoder, pencere seçimi, seed ve eğitim bütçesi değiştirilmez.

## Önceden belirlenen yöntem

`configs/transformer_weighted.json`, başlangıç ModernBERT config'iyle aynıdır;
yalnız `positive_weighting="sqrt_capped"`, `positive_weighting_cap=50` ekler.
İlk eğitim gibi üç epoch, FP32/SDPA, batch 1, accumulation 8, lr 2e-5 ve seed
42 kullanır. Daha önceki modelden devam edilmez; aynı pinned pretrained
başlangıçtan yeni classifier ile başlar.

Her türün BCE pozitif terimi şu katsayıyla çarpılır:

`min(50, max(1, sqrt(negative_response_tokens / positive_response_tokens)))`

- Sayım **yalnız train** kayıtlarının eğitimde seçilen pencerelerinden yapılır.
  Kaynak, özel token ve padding, `response_mask & attention_mask` ile dışlanır.
- Dört tür bağımsız sayılır. Örtüşen hedef aynı tokenı birden fazla türde
  pozitif yapabilir; etiketler tek sınıfa zorlanmaz.
- Sayım bir streaming geçiştir; bütün veri için token/target tensörleri RAM'e
  toplanmaz. `WindowDataset` doğrudan kullanılır; DataLoader base-seed çekimi
  yapılmaz ve Torch RNG/model-head başlangıcı değiştirilmez.
- Hiç pozitif token olmayan bir tür için sonlu cap kullanılır. Yeni pozitif
  hedef üretilmez. Sayılacak geçerli cevap tokenı yoksa eğitim durur.
- Sayılar span/karakter sayıları değil, mevcut loss'un gördüğü **cevap tokenı
  tekrarlarıdır**. Uzun cevap parçaları örtüşürse tekrarlar da loss gibi sayılır.
- Validation/test sınıf oranlarından ağırlık hesaplanmaz; validation yalnız
  checkpoint seçimi ve daha sonra önceden sınırlandırılmış eşik seçimi içindir.

## Kayıt ve checkpoint

`positive_weighting.json` yöntem, cap, dört türün pozitif/negatif token
sayıları, ağırlıklar, maske/pencere politikası ve sayım süresini kaydeder.
Aynı bilgi `transformer_summary.json` ve checkpoint metadata'sında bulunur;
etkin config hesaplanan `positive_weights` vektörünü de saklar.

Modelde vektör, config'ten geri yüklenen ve `.to(device)` ile taşınan bir
buffer'dır; eski head safetensors dosyasının alanları değişmez. Eski checkpoint
ve config'lerde weighting yoksa önceki normal BCE davranışı sürer.

Validation BCE'de **aynı train-derived vektör** kullanılır. Checkpoint,
bu deney içindeki en düşük ağırlıklı validation BCE ile seçilir; eşitlikte
ilk epoch korunur. Ağırlıklı ve ağırlıksız BCE büyüklükleri model üstünlüğü
için doğrudan karşılaştırılmaz. Modellerin karşılaştırması ortak F1 ile yapılır.

Tahmin logits'leri veya sigmoid skorları için prior düzeltmesi yapılmaz.
Ağırlıklandırma, loss ve öğrenilen ağırlıklar üzerinden etki eder. Karar eşikleri
ayrı validation artifact'iyle seçilir; mevcut model/calibration dosyaları korunur.

## Önceden belirlenen ikinci aşama

`configs/calibration_weighted_policy.json` eğitimden önce şu aramayı ilan eder:

`[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]`

Eğitim bitince bu policy'den, kaydedilmiş iyileşen **ağırlıklı validation BCE**
checkpoint'lerinin tamamı epoch sırasıyla alınarak gerçek config üretilir:
en fazla üç aday. Kaydedilmemiş epoch yeniden üretilmez. Eşik/aday seçim kuralları ilk calibration ile aynı:
tür başına character F1, dört tür macro F1, ardından any F1 ve config sırası.
Policy'de henüz koşum yolları yoktur; doğrudan kalibrasyon komutu değildir.
Model aileleri de validation typed macro F1, ardından any F1 ile karşılaştırılır;
tam eşitlikte orijinal ağırlıksız model tercih edilir.
Model ailesi seçimi ve bütün ayarlar resmî testten önce sabitlenir.

## Merkezî çalıştırma

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_CACHE=.cache/hf-hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/run_job.py configs/jobs/transformer_weighted_train.json
```

Bu belge kod/deney reçetesini açıklar; tam eğitimin çalıştırıldığı iddiası
değildir. Job, calibration policy'sini de checksum girdisi olarak kaydeder.
Resmî test veya target politikasında değişiklik yoktur.

## Kontroller

`python -m unittest discover -s tests -p test_positive_weighting.py -v`:
formül/floor/cap, train-only guard, streaming ve RNG korunması, örtüşen hedefler,
source/padding gradient maskesi, normal BCE eşdeğerliği, checkpoint reload ve
inference logits'lerine prior düzeltmesi yapılmaması test edilir. Küçük CPU CLI
testi sayımın yalnız train kayıtlarıyla çağrıldığını ve kayıtlı ağırlıkların
validation/checkpoint boyunca aynı kaldığını doğrular.
