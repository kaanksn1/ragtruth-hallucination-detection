# Transformer merkezî incelemesi — 31 Ekim 2026

İncelenen ilk teslim: 4. kişinin PR #4'ü, `ea358486f530271b6c921011c6413fe562c81f38`.
Modelin ve ilk testlerin yazarlığı 4. kişiye aittir. Kullanıcının açık
atamasıyla repo sahibi entegrasyon hatalarını ayrı commit'te düzeltir.
Bu inceleme kişisel deney analizi veya sunum teslimi değildir.

## Bulgular ve değişiklikler

- **Validation/veri kimliği:** Loader dosyanın güncel summary'ye uymasını
  denetliyordu, checkpoint'in eğitimde kullandığı veriyle karşılaştırmıyordu.
  Değiştirilmiş validation + ona uygun yeniden yazılmış summary kabul
  edilebiliyordu. Şimdi validation/manifest/summary hash'leri checkpoint
  metadata'sıyla da eşleşir. Dosya taşındığında kimlik değişmesin diye
  bilgisayar yolları yerine sabit rol adları kaydedilir.
- **Donmuş test girdileri:** Test kapısı veri ve checkpoint dosyalarını
  kapsıyor, çalışan model/CLI/ortak hizalama kodunu zorunlu tutmuyordu.
  Şimdi bunlar ve requirements açıkça hash girdileri arasında aranır;
  eksik beyanda test içeriği/ağırlıkları açılmadan hata verilir. Boş bir
  checkpoint klasörü `checkpoint.json` şartını da atlayamaz.
- **Sabit revision:** `main` gibi zamanla değişen Hub revision'ları kabul
  edilmez; 40 haneli commit SHA gerekir. Mevcut model revision'ı değişmedi.
- **Ortam:** Requirements Transformers 4.57.1 isterken repo sahibinin mevcut
  Windows/ROCm ortamlarında 4.57.6 vardı. 4.57.6 ile kontroller geçti ve
  requirements buna sabitlendi. Hiçbir PyTorch/ROCm paketi değiştirilmedi.
- **Pilot/ölçüm:** `--epochs` pozitif CLI sınırı ve `transformer_pilot.json`
  eklendi: tek epoch, 8 train / 4 validation, GPU zorunlu, merkezî kayıt.
  Train/validation bölümleri senkronize süre, cevap-token hızı ve GPU allocator
  peak allocated/reserved byte değerlerini summary'ye ekler. Bunlar tüm
  sistemin VRAM kullanımı veya tek başına tam eğitim tahmini değildir.

Yeni regresyonlar ilk teslimde veri/checkpoint eşlemesi, taşınabilir kimlik,
sabit revision ve test girdileri kontrolündeki eksikleri gösterdi. Düzeltilmiş
kodda geçer; ek tam-girdi testi doğru donmuş dosya listesini kabul eder,
eksik/değiştirilmiş kodu reddeder. Gerçek resmî test verisi kullanılmadı.

## Doğrulama

- **101 test senaryosu:** Windows HF/PyTorch ortamında 99 geçti, iki native
  CRF testi paket bulunmadığından atlandı. Ayrı P2 CRF ortamında 79 geçti,
  22 HF/PyTorch testi paket bulunmadığından atlandı. Her koşullu test en az
  bir ortamda çalıştı; atlamalar başarılı test sayılmadı.
- **26 Transformer testi:** Altı yeni regresyon dahil, gerçek sabit
  ModernBERT tokenizer'ıyla tamamı Windows CPU'da geçti. Tokenizer için dört
  küçük JSON indirildi; pretrained encoder ağırlıkları indirilmedi.
- **GPU kod kontrolü:** Mevcut WSL/ROCm ortamında RX 7800 XT, iki katmanlı
  rastgele ModernBERT, FP32/SDPA, gradient checkpointing, batch 1 × 2048;
  bir AdamW adımı ve validation ileri geçişi geçti. Sonlu loss ve değişen
  sınıflandırıcı ağırlıkları doğrulandı. Bu tam ModernBERT-base pilotu veya
  gerçek veride başarı deneyi değildir.
- Pilot ve full-train job'ları `--check-only` ile doğrulandı; çalıştırılmadı.
  Ham/işlenmiş veri, CRF kodu ve split manifesti değiştirilmedi.

Yerel kayıtlar Git dışında: `.cache/validation/transformer_hf_review.log`,
`.cache/validation/transformer_crf_review.log`, `outputs/transformer_gpu_review.json`.
4. kişinin kendi ortamında bildirdiği pretrained CPU kontrolü tarihsel olarak
[TRANSFORMER_REVIEW.md](TRANSFORMER_REVIEW.md) içinde korunur; burada yeniden
pretrained model denendiği iddia edilmez.

## Sonraki merkezî koşum

Paket kurulumu gerekmez; mevcut WSL ortamı requirements ile uyumludur.
İlk gerçek pretrained pilot için model ağırlıklarının sabit revision'ı
önceden indirilmelidir. Ardından depo kökünde:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/run_job.py configs/jobs/transformer_pilot.json
```

Pilotun 8/4 ilk kaydı bütün uzunluk dağılımını temsil etmez. Sonraki tam
eğitimden önce gerçek encoder bellek/hız kaydı ve uygun bütçe belirlenmeli.
Şu an checkpoint seçimi minimum validation BCE, eşik sabit 0.5'tir. Ortak
scorer ve olası sınırlı validation eşik/F1 kararı 4–5. kişilerle tamamlanmalı;
resmî testle seçim yapılmaz. `best` eğitim / tüm pencereler + `min` tahmin
sezgiseli değişmedi; yanlış negatif riski nihai analizde açıklanmalıdır.
Tam eğitim, gerçek sonuç/hata örnekleri ve 4. kişinin slaytları bekliyor.
