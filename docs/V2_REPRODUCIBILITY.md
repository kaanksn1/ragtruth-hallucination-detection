# V2 çalıştırma ve tekrar üretilebilirlik

V2 tamamlandı. [Sonuç raporu](../results/v2/FINAL_REPORT.md),
[denetim](../results/v2/AUDIT_TR.md) ve [dondurulmuş karar](V2_FINAL_PROTOCOL.json)
güncel başvuru noktalarıdır. Eski V1 raporları tarihsel resmî test sonuçlarıdır.
Bu belgedeki komutlar yeniden çalıştırma rehberidir; denetim sırasında yeni
eğitim, ayar araması veya yeni test tahmini yapılmadı.

## Hafif teslim ve yerel dosyalar

Denetlenmiş ZIP'in kökündeki `demo.html` doğrudan tarayıcıda açılır. Kaydedilmiş
2.240 doğrulama cevabını ve üç V2 modelinin tahminlerini gösterir. Canlı model
çalıştırmaz. İnternet, Python ve GPU gerektirmez.

ZIP kod, test, yapılandırma, split manifestleri ve raporları içerir. Tam veri,
eğitilmiş ağırlıklar, ortamlar, Git geçmişi ve tüm tahmin dosyaları içermez.
Dolayısıyla arşivi açmak tek başına tam eğitimi/değerlendirmeyi tekrar üretmez.
GitHub da ağırlıkların yedeği değildir. Yerelde korunması gerekenler:

- `data/processed/`, `data/splits/`: V2 veri ve ayrım kimlikleri.
- `outputs/v2_study_v2.json`, `outputs/runs/`: dokuz epoch checkpoint'i, çalışma
  kayıtları, calibration JSON'ları, olasılık önbellekleri ve sabit tahminler.
- `docs/V2_FINAL_PROTOCOL.json`, `configs/v2/final/`: hangi dosyanın hangi SHA
  ile kullanıldığını belirleyen commit edilmiş protokol/job çiftleri.

Seçilen üç checkpoint'in dosya SHA'ları protokolde; dokuz checkpoint'in
manifest yolları ve denetim sonuçları
[`integrity_audit.json`](../results/v2/integrity_audit.json) içindedir.

## Ortam

Gerçek eğitim: Windows üzerinde Ubuntu-24.04 WSL, RX 7800 XT 16 GB,
Python 3.12, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, Transformers 4.57.6,
tokenizers 0.22.2, FP32. Kurulu ortam `/opt/banking77-venv/bin/python` idi.
Bu yol kullanıcıya özeldir. `scripts/run_v2_study.py` de bu makinenin WSL,
worktree ve ortam yollarını kullanır; başka bilgisayarda aynen çalışacak genel
bir kurulum programı değildir. GPU sürücüsü/PyTorch kurulumu bu hafif ZIP'te yoktur.

Uyumlu PyTorch ortamında `requirements-transformer.txt`, baseline testleri için
`requirements-crf.txt`, grafik üretmek için `requirements-evaluation.txt`
kullanılır. Gereksiz yere GPU'ya uygun mevcut PyTorch yerine başka build kurmayın.
Ortamdan sonra proje kökünde `python -m unittest discover -s tests -v` çalışır.
Torch veya python-crfsuite yoksa ilgili testlerin atlanması tam doğrulama sayılmaz;
GitHub'daki Transformer kontrolü her iki bağımlılığı birlikte kurar.

Encoder `answerdotai/ModernBERT-base`, revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`.
İlk pretrained `model.safetensors` SHA-256 değeri
`340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.
Model/tokenizer normal eğitim CLI'sinde bu revision ile yüklenir. Offline
çalıştırmadan önce aynı snapshot ilgili Hugging Face cache'inde bulunmalıdır.
Yerel çalışmada `HF_HUB_CACHE=.cache/hf-hub`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HSA_ENABLE_DXG_DETECTION=1` kullanıldı.
Yeni makinede bu offline değişkenlerini boş cache ile açmak indirmeyi engeller.
V1 task-finetuned checkpoint'i V2 başlangıcı olarak kullanılamaz.

## Veri yolları

`scripts/prepare_v2_data.py` varsayılan kaynak yolunu sahibin iç içe worktree
düzeninden çıkarır. Başka yerleşimde **açık kaynak kökü** verin. Kaynak kökte
orijinal projenin hazırlanmış `data/processed/train.jsonl`, `validation.jsonl`
ve `data/splits/split_manifest_v1.csv` dosyaları bulunmalı. Bu girdiler V1
indirme ve veri hazırlama adımlarıyla, `--include-test` vermeden hazırlanır;
orijinal resmî test anotasyonları V2'de açılmaz.

```text
python scripts/prepare_v2_data.py --source-root /path/to/original-project
```

Bu komut V2 kökünde train/dev dosyalarını üretir; kaynak projenin dosyalarını
değiştirmez. Yeni holdout yalnız V2 kararları Git'te dondurulduktan sonra ayrı
`--include-holdout` adımıyla hazırlanmıştır. Mevcut sonuçları incelemek için
bu adımı veya sabit test job'larını yeniden çalıştırmaya gerek yoktur.

## Eğitim ve doğrulama komutları

Bağımlılıkları ve verisi hazırlanmış bağımsız bir yeniden üretim kopyasında:

```text
python scripts/transformer.py train --config configs/v2/bce.json --output outputs/reproduction/bce --device cuda
python scripts/transformer.py train --config configs/v2/focal.json --output outputs/reproduction/focal --device cuda
python scripts/structured_transformer.py train --config configs/v2/structured.json --output outputs/reproduction/structured --device cuda
```

ROCm da PyTorch CLI'sinde `cuda` cihaz adını kullanır. Çıktı dizinleri yeni/boş
olmalı. Gerçek çalışma her ailede üç epoch ve bütün iyi-kaliteli train/dev
cevaplarını kullandı; küçük limitli pilotlar nihai skor değildir.

BCE/focal calibration, `scripts/calibrate_transformer.py --config <JSON>
--output <yeni-dizin> --device cuda` komutuyla yapılır. JSON'un şeması ve gerçek
15-değerli eşik grid'i `results/v2/{bce,focal}_calibration.json` içindeki `config`
alanında korunur; `checkpoints` alanı yeniden eğitimin epoch-1/2/3 yollarına
uyarlanmalıdır. Varsayılan `configs/calibration.json` V1'e aittir ve V2 için
kullanılmaz. Structured için `scripts/structured_transformer.py calibrate
--checkpoints <epoch-1> <epoch-2> <epoch-3> --output <yeni-dizin> --device cuda`
kullanılır. Seçim kuralları [V2_PLAN.md](V2_PLAN.md)'de sabittir.

Yalnız dev tahmini için her iki predict CLI'sine de `--split validation`,
`--processed-dir data/processed`, `--manifest data/splits/split_manifest_v2.csv`,
`--split-summary data/splits/split_summary_v2.json`, `--checkpoint <seçilen-yol>`,
`--calibration <calibration.json>` ve `--output <yeni-jsonl>` açıkça verilmelidir.
Structured CLI'nin varsayılan processed yolu `data/processed_v2`; BCE/focal
CLI'sinin varsayılan manifesti V1'dir. Merkezi job'lar doğru V2 yollarını açıkça
verdi; varsayılanlara güvenilmemelidir.

## Tekrar üretimin sınırları ve denetim

Seed 42 ve konfigürasyonlar sabittir; tek seed ile eğitim yapıldı. Farklı sürücü,
donanım veya paket build'lerinde bit düzeyinde aynı ağırlık/skor garantisi yoktur.
Kaydedilen epoch checkpoint'leri inference için yeterlidir; optimizer/RNG ve
yarım epoch konumu kaydedilmediğinden kesilen eğitime tam kaldığı adımdan devam
edilemez. Supervisor'ın tamamlanmış aşamalar için genel resume CLI'si de yoktur.
Mevcut çalışmada bütün aşamalar tamamlandı; bu kısıt sonuç dosyalarını silmez.

Yerel orijinal train/dev dosyaları ve tamamlanmış V2 çıktıları mevcutsa:

```text
python scripts/audit_v2_integrity.py --original-root /path/to/original-project --output outputs/new-integrity-audit.json
python scripts/package_v2_handoff.py --output artifacts/new-audited-v2.zip
```

İlk komut yalnız orijinal train/dev'yi ve V2 dosyalarını okur; orijinal resmî
testi açmaz. Sabit V2 tahminlerinin metriklerini bağımsız karakter kümeleriyle
doğrular; model veya ayar seçmez. Her iki komut mevcut çıktı üzerine yazmayı
reddeder. Arşiv, yol güvenliği, CRC ve SHA-256 manifest kontrolünden geçer.
