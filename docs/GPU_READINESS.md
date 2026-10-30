# GPU hazırlığı — 29 Ekim 2026

Durum: **Sentetik donanım kontrolü geçti. Gerçek Transformer GPU pilotu
henüz yapılmadı.** Bu kontrol repo sahibinin donanım/koşum sorumluluğudur;
2. kişinin veri hazırlığını veya 4. kişinin model uygulamasını tamamlamaz.

31 Ekim'de gerçek verinin küçük bir bölümüyle yapılan CRF CPU doğrulaması
[BASELINE_CRF.md](BASELINE_CRF.md) içinde kayıtlıdır; GPU modeli/pilotu değildir.

## Yapılan işlem

Mevcut WSL/PyTorch ortamında, varsayılan DistilBERT boyutlarında rastgele
başlatılmış bir encoder ve tek kullanımlık dört kanallı doğrusal başlıkla kısa
forward/backward/AdamW adımları çalıştırıldı. Ağırlık veya tokenizer indirilmedi.
Hiçbir RAGTruth kaydı, etiketi veya resmî test içeriği kullanılmadı.

- GPU: AMD Radeon RX 7800 XT; görünen toplam VRAM yaklaşık 15,8 GiB.
- Python 3.12.3, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, Transformers 4.57.6.
- Geçici mimari: 6 katman, 768 hidden, 12 attention head; 66.365.956 parametre.
- FP32, eager attention, batch 2; 2 ısınma optimizer adımı + 5 ölçüm adımı.
- Girdiler rastgele token ID'leri; hedefler dört bağımsız sentetik ikili kanal.
- Kayıp yalnız sentetik cevap konumlarında hesaplandı; kaynak/padding
  konumlarının loss'a katılmadığı kontrol edildi. Encoder'ın kaynak bağlamına
  doğru gradient taşıması bununla çelişmez.
- Sonlu warmup gradientleri, sonlu loss ve optimizer'ın başlık ağırlıklarını
  değiştirdiği doğrulandı. Ölçüm sınırlarında GPU senkronizasyonu yapıldı.

## Ölçümler

| Sentetik girdi uzunluğu | Batch | Ölçülen adım | Saniye/adım | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| 256 token | 2 | 5 | 0,0392 | 1,384 GiB | 1,549 GiB |
| 512 token | 2 | 5 | 0,0722 | 1,595 GiB | 1,803 GiB |

`Allocated/reserved` PyTorch'un bu süreç için kaydettiği allocator değerleridir;
ekran sürücüsü ve diğer uygulamalar dahil toplam GPU kullanımı değildir.
Ölçüm aralığı çok kısadır ve yalnız bu sabit sentetik şekilleri kapsar.

Bu sayıların gerçek eğitim hızı olduğu, bütün verinin 512 tokenda temsil
edilebildiği veya nihai modelin aynı belleğe sığacağı sonucuna varılmaz.
Tokenizasyon, gerçek padding/uzunluk dağılımı, kaynak pencereleri, tahmin
birleştirme, değerlendirme ve checkpoint yazımı ölçülmedi. **Bu hızdan epoch
veya tam eğitim süresi tahmini üretilmedi.**

Geçici donanım başlığı projenin nihai dört tür/sınır/örtüşme kodlaması değildir.
Model mimarisi ve decoding kararı 2/4. kişilerin ortak sözleşmesiyle belirlenecek.

## Kayıtlar ve tekrar çalıştırma

Kod: [gpu_smoke.py](../scripts/gpu_smoke.py).

Makine çıktıları yerel ve Git dışında:

- `outputs/gpu_smoke_256.json`
- `outputs/gpu_smoke_512.json`

Mevcut ortamda, depo kökünden:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/gpu_smoke.py --sequence-length 256 --batch-size 2 --warmup-steps 2 --steps 5
```

512 token için yalnız `--sequence-length 512` değiştirilir. Varsayılan çıktı
adında UTC zaman damgası bulunur. Mevcut bir raporun üzerine yazılmaz. GPU
zorunludur; CPU'ya sessiz geçiş yapılmaz. Paketler veya mevcut BANKING77
checkpoint/çıktıları bu kontrol sırasında değiştirilmedi.

Kaynak API açıklamaları:
[DistilBERT konfigürasyonu](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/distilbert).

## İlk gerçek pilot için gereken teslimler

1. **2. kişi:** Kaynak grupları korunmuş train/validation manifesti, kaynak–cevap
   serileştirmesi, tokenizer/offset eşlemesi, loss maskeleri ve pencere eşlemesi.
2. **4. kişi:** Nihai amaçla uyumlu model/eğitim kodu, tokenizer/model kimliği,
   örtüşen hedef/decoding tasarımı ve kısa pilot konfigürasyonu.
3. **5. kişi:** Ortak değerlendirme arayüzü ve boş/örtüşen/sınır hataları için
   doğrulanmış metrik davranışı.
4. **1. kişi:** Bu teslimlerle yalnız eğitimden alınmış küçük bir örnekte
   gerçek pilotu çalıştırır; süre, pencere sayısı ve bellek kaydından eğitim
   bütçesini hesaplar. Resmî test kullanılmaz.

3. kişinin CRF kodu hazır olduğunda onun tam eğitim koşumu da 1. kişinin
bilgisayarında, aynı ortak veri sözleşmesiyle çalıştırılır.
