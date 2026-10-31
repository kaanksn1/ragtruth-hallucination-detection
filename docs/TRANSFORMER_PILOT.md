# Gerçek model GPU pilotu — 31 Ekim 2026

Repo sahibinin RX 7800 XT bilgisayarında pretrained ModernBERT-base pilotu
tamamlandı. Bu tam eğitim/başarı raporu değildir; resmî test kullanılmadı.

## Kimlik ve ortam

WSL Ubuntu-24.04, Python 3.12.3, Torch `2.9.1+rocm7.2.1.gitff65f5bc`,
Transformers 4.57.6. Mevcut paketler değiştirilmedi. FP32/SDPA, batch 1,
gradient accumulation 8 ve gradient checkpointing kullanıldı.

Model/tokenizer `answerdotai/ModernBERT-base`, revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. İndirilen `model.safetensors` SHA-256:
`340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.
Önbellek proje içindeki `.cache/hf-hub/` klasörüdür; Git dışındadır.

## İlk gerçek pilot ve checkpoint kontrolü

Temiz `44b7aa1`, `configs/jobs/transformer_pilot.json`: 8 train / 4 validation,
tek epoch, bir AdamW adımı. Eğitim 2,66 saniye, validation loss 0,14 saniye;
yükleme/kaydetme/yeniden yükleme/tahmin dahil süreç 72,79 saniye sürdü.
İndirme bu süreye dahil değildir. Eğitim peak allocated yaklaşık 3,00 GiB.
İlk kısa örneklerden tam veri süresi veya maksimum bellek çıkarılmadı.

Checkpoint'in yeniden yüklemesinde otomatik derleme açılıyordu. Kurulu HF
`ModernBertConfig.to_dict`, `reference_compile` alanını kasıtlı olarak atıyor.
Loader artık encoder oluşturulmadan bu alanı tekrar `False` yapar. Gerçek
checkpoint düzeltilen yükleyiciyle GPU'da yeniden açıldı; dört validation
tahmini önceki JSONL dosyasıyla bayt bayt aynı. Regresyon testi serializer'ın
alanı atmasını ve loader'ın doğru değeri geri koymasını kapsar.

Yerel kayıtlar:

- `outputs/runs/20261031T105255989490Z-transformer-pilot/`
- `outputs/runs/20261031T105910295074Z-transformer-pilot-reproduce/`

## Uzunluğa göre ölçüm ve süre projeksiyonu

`scripts/profile_transformer_runtime.py`, temiz `bcd87f3` ile 12.702 good train
ve 2.240 good validation cevabını saydı. Split değişmedi. Sabit seed'li
reservoir seçimi yalnız girdilerin uzunluğunu kullanır; gold yalnız örneklenen
kayıtların mevcut loss hesabında kullanılır. Her grup 8'e kadar gerçek pencere
ve iki tekrar, toplam 16 pencere ile ölçüldü. Isınma ölçüm dışıdır.

| Uzunluk | Train pencere | Validation pencere | Train sn/pencere | Validation sn/pencere |
|---|---:|---:|---:|---:|
| 1–256 | 61 | 7 | 0,065 | 0,017 |
| 257–512 | 2.428 | 434 | 0,093 | 0,024 |
| 513–768 | 4.237 | 734 | 0,127 | 0,038 |
| 769–1024 | 3.179 | 621 | 0,188 | 0,053 |
| 1025–1536 | 2.327 | 352 | 0,320 | 0,100 |
| 1537–2048 | 470 | 122 | 0,622 | 0,199 |

Gerçek en uzun pencere örneklemeye eklendi: **2048 token çalıştı**.
En yüksek allocated **3,37 GiB**, reserved **5,74 GiB**; bunlar sürecin
allocator değerleridir, toplam sistem VRAM kullanımı değildir.

Pencere sayılarıyla ağırlıklı projeksiyon: epoch başına train **40,0 dakika**,
validation loss **2,2 dakika**. Üç epoch merkez tahmin **2,11 saat**.
Yükleme/checkpoint/final JSON decoding ve sistem yükü için pratik planlama
aralığı **2–3,5 saat**. Tam epoch henüz ölçülmedi; bu bir bitiş garantisi veya
istatistiksel güven aralığı değildir.

Profil ağırlıkları kaydedilmedi ve tam eğitime aktarılmaz. Tam eğitim pinned
başlangıç ağırlıklarından seed 42 ile yeniden başlar. Profil loss'ları başarı
metriği değildir. Yerel kayıt:
`outputs/runs/20261031T105659713838Z-transformer-runtime-profile/`.
Girdi/model hash'leri `run.json`, ölçümler `runtime_profile.json` içindedir.

## Tam eğitim komutu

Önbellek hazırken depo kökünde:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_CACHE=.cache/hf-hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/run_job.py configs/jobs/transformer_train.json
```

Üç epoch, 12.702 train / 2.240 validation. Epoch başlangıcı ve her 500 pencere
console.log'a yazılır. Checkpoint minimum validation BCE ile seçilir; eşik
0.5'tir. Nihai F1/hata analizi ortak scorer ile eğitim sonrasında yapılır;
resmî test protokolü hâlâ taslaktır.
