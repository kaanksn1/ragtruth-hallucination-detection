# Merkezî koşum — Serdar'ın bölümü

Güncel durum: tam eğitimler, validation seçimi ve dondurulmuş ayarlarla resmî test
değerlendirmesi tamamlandı. [Teknik rapor](../results/FINAL_REPORT.md),
[nihai karar](FINAL_PROTOCOL.json) ve `configs/final/` altındaki commit edilmiş
job/protokol çiftleri güncel teslimdir. Aşağıdaki tarihli ilk kontroller,
yapıldıkları tarihteki durumu kaydeder.

Bu katman yalnız veri indirme/doğrulama, komut çalıştırma ve kayıt tutmayı
yapar. CRF, Transformer, veri hizalama ve ortak scorer ilgili üyelerden gelir.
Tam model eğitimi henüz başlamadı. Nihai test protokolü hâlâ taslak.

## Veri ve testler

Python 3.12 ve Git yeterlidir; bu iki komut için ML paketi gerekmez:

```powershell
python scripts/download_data.py
python scripts/download_data.py --check-only
python -m unittest discover -s tests -v
```

İndirme, `configs/source.json`daki sabit upstream commit'ini ve SHA-256'ları
kullanır. Mevcut doğru dosyayı yeniden indirmez; yanlış dosyanın üstüne yazmaz.
`--check-only` ağa çıkmaz. Ham veriyi parse etmez veya test etiketini incelemez.
Ham veri ve upstream LICENSE yerel `data/raw/`dadır ve Git dışında kalır.

## Bizim GPU ortamında kayıtlı sentetik kontrol

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/run_job.py configs/jobs/gpu_smoke_256.json
```

Bu makineye özel mevcut ortam yeniden kullanılır; paket kurulmaz/güncellenmez.
Diğer üyeler bu WSL yolunu kopyalamak zorunda değildir. GPU scripti ihtiyaçları
kurulu PyTorch ve Transformers'tır; gerçek model bağımlılıkları model PR'ında
bildirilir. Bu sentetik kontrolün çıktısı başarı skoru veya eğitim süresi
tahmini değildir; kapsamı [GPU_READINESS.md](GPU_READINESS.md) açıklar.

`outputs/runs/<UTC-zamanı>-<ad>/` içinde:

- `run.json`: config kopyası/hash'i, gerçek argv, seed, commit, branch, kirli
  dosya listesi, girdi hash'leri, Python/paket sürümleri, seçilmiş GPU/offline
  ortam değişkenleri, süre ve exit code. Kimlik bilgileri kaydedilmez.
- `console.log`: stdout ve stderr; başarısızlıkta da korunur.
- Job'ın kendi çıktı dosyaları ve bunların hash'leri.

`seed` kaydı tek başına determinizm sağlamaz; model komutu bu seed'i kullanmalı.
Çocuk sürecin GPU/zaman ölçümü gerekiyorsa model scripti de onu raporlamalı.
Wrapper bütün sistemin VRAM'ini veya çocuk sürecin örnek hızını ölçmez.

## Yeni model teslimini sıraya alma

Model sahibi şu biçimde bir config verir; bu örnek henüz mevcut olmayan bir
dosya/CLI için sözleşmeyi gösterir, çalıştırılabilir model olduğu iddiası değildir:

```json
{
  "schema_version": 1,
  "name": "crf-validation",
  "stage": "train",
  "split": "train",
  "seed": 42,
  "inputs": ["data/processed/manifest.json", "configs/crf.json"],
  "command": ["{python}", "train_crf.py", "--seed", "{seed}", "--output", "{run_dir}"],
  "expected_outputs": ["validation_predictions.jsonl"],
  "timeout_seconds": 3600
}
```

Yer tutucular yalnız `{python}`, `{seed}`, `{run_dir}`. Komut shell metni değil
argv listesi olarak doğrudan çalışır; veri/checkpoint/manifest ve komutun
kullandığı tüm config dosyaları `inputs` listesine eklenir. Girdiler depo
içinde olmalıdır. Çıktı yolları o koşum klasörüne göredir. Beklenen dosya eksikse
exit code sıfır bile olsa koşum başarısızdır.

```powershell
python scripts/run_job.py configs/jobs/crf_validation.json --check-only
python scripts/run_job.py configs/jobs/crf_validation.json configs/jobs/transformer_train.json
```

Komutlar sırayla çalışır; ilk hata sonraki işi durdurur. `outputs/.run.lock`
iki wrapper'ın aynı anda çalışmasını engeller. Süre sınırında veya hatada log
korunur. İşletim sistemi süreci zorla kapatırsa kilit kalabilir; başka bir koşum
olmadığı doğrulanmadan kilidi kaldırmayın. Wrapper dışından açılmış süreçler
bu kilide dahil değildir.

## Resmî test kapısı

Pilot/train için `split: test` kabul edilmez. `predict`/`evaluate` test
koşumu ancak temiz çalışma ağacında ve commit edilmiş bir JSON protokolüyle
başlayabilir. Önce [bilimsel protokol](PROTOCOL_DRAFT.md) tamamlanır.

Test job config'ine `"frozen_protocol": "configs/final_protocol.json"` eklenir.
Protokol, deney kararlarının yanında aşağıdaki makine alanlarını içerir:

```json
{
  "status": "frozen",
  "test_config_sha256": "exact-test-job-config-sha256",
  "inputs_sha256": {"relative/input/path": "exact-file-sha256"}
}
```

Bu alanlar örnektir; mevcut bir test izni değildir. `inputs_sha256` job'ın tüm
girdileriyle tam eşleşmeli. Nihai checkpoint, model/scorer kodları, manifest ve
ön işleme/eşik/config dosyaları da girdilere dahil edilmeli. Protokol ve job
config'i testten **önce commit edilir**. Wrapper protokolün commit'ini ve
hash'ini kaydeder; farklı config veya girdi baytlarıyla koşumu reddeder.

Bu teknik kapı beyan edilen dosyaları doğrular; bilimsel kararların uygunluğunu
ve beyan edilmeyen veri kullanımını otomatik denetlemez. Kod incelemesi gerekir.
Testten sonra yeni ayar seçilmez; test tekrarları otomatik başlatılmaz.

## Doğrulama — 30 Ekim 2026

- Windows Python 3.12: 12 birim/entegrasyon kontrolü geçti. Başarısız çocuk
  sürecin exit code/logu, eksik çıktı, paralel kilit ve test protokolü kontrol edildi.
- Yerel üç ham dosyanın SHA-256'sı sabit kaynak manifestiyle eşleşti.
- `add9af7` commit'inde temiz çalışma ağacıyla WSL/RX 7800 XT sentetik job'ı
  exit code 0 ile tamamlandı. Config, commit, paketler, çıktı hash'i ve log
  `outputs/runs/20261030T081915123902Z-gpu-smoke-256/` altında kaydedildi.
- Bu job gerçek RAGTruth verisini veya test etiketlerini kullanmadı. İlk gerçek
  model pilotu hâlâ veri/model teslimlerini bekliyor.
- Mevcut WSL ortamında `sklearn-crfsuite` ve `python-crfsuite` kurulu değil;
  CRF sahibinin bağımlılık tesliminden sonra gerekli kurulum ayrıca yapılacak.

## CRF merkezî doğrulaması — 31 Ekim 2026

CRF için proje kökünde ayrı bir Windows `.venv` oluşturuldu ve yalnız
`python-crfsuite==0.9.12` kuruldu. BANKING77/PyTorch ortamının paketleri değişmedi.

```powershell
& '.venv/Scripts/python.exe' scripts/run_job.py configs/jobs/crf_smoke.json
& '.venv/Scripts/python.exe' scripts/run_job.py configs/jobs/crf_validation.json --check-only
```

İlk komut gerçek eğitim verisinin 300, validation'ın 100 cevabıyla kısa
kontroldür. İkinci komut tam eğitim reçetesini **çalıştırmadan** doğrular.
Tam eğitim için `--check-only` kaldırılır; bu tam koşum henüz yapılmadı.
Kısa koşumun kaydı ve tekrar üretim sonucu [BASELINE_CRF.md](BASELINE_CRF.md)
içindedir. Resmî test ve gerçek Transformer GPU pilotu yapılmadı.

`run_job`, doğrulanmış çocuk sürece `RAGTRUTH_RUN_RECORD` yolunu verir. CRF'nin
resmî test çağrısı bu kayıtta aynı komut ve donmuş protokol bulunduğunu tekrar
doğrular. Bu kayıt, daha önce tanımlanan protokolü ve gerçek çalıştırma komutunu eşleştirir.

## Transformer tesliminin merkezî incelemesi — 31 Ekim 2026

4. kişinin model/CLI teslimi incelendi; veri kimliği ve donmuş test girdileri
kontrolü düzeltildi. 26 Transformer testi, gerçek tokenizer hizalaması ve
küçük rastgele ModernBERT ile 2048-token RX 7800 XT kod kontrolü geçti.
Bu GPU kontrolü pretrained encoder belleğini veya tam eğitim hızını ölçmez.
Mevcut ROCm ortamının paketleri değiştirilmedi; requirements 4.57.6 ile uyumlu.

Pretrained gerçek pilot job'ı hazır ve henüz çalıştırılmadı; önce sabit
revision'ın model ağırlıkları indirilmeli. Sonraki komut ve sonuçların
sınırları [TRANSFORMER_INTEGRATION.md](TRANSFORMER_INTEGRATION.md) içinde.
Bu ilk inceleme sırasında tam eğitim ve resmî test başlatılmamıştı.
Sonraki gerçek pretrained pilot ve uzunluk profili tamamlandı:
[ölçümler ve tam eğitim komutu](TRANSFORMER_PILOT.md). Ortak scorer 5. kişiden bekleniyor.
