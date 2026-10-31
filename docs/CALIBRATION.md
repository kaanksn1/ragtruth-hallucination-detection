# Validation ile tür başına eşik seçimi

Bu deney mevcut ModernBERT ağırlıklarını yeniden eğitmez. İlk koşumun sabit
0.5 eşiğini karşılaştırma olarak korur; ayrı bir decoder artifact'i üretir.
Model checkpoint'i, ham anotasyonlar ve orijinal tahmin dosyaları değiştirilmez.

## Önceden sınırlandırılmış arama

`configs/calibration.json`, gerçek skor önbelleği oluşturulmadan önce ilan
edildi: aynı tamamlanmış train koşumunun kaydedilmiş **epoch-1 ve epoch-2**
checkpoint'leri; her tür için eşikler
`[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]`.

- Her tür kendi validation character F1'ını en yüksek yapan eşiği seçer.
  Tam eşitlikte daha yüksek eşik tercih edilir.
- Bu 11^4 birleşimin aranması değildir; dört bağımsız 11 değerlik aramadır.
- İki checkpoint, seçilmiş eşikleri uygulandıktan sonraki **dört türün macro
  character F1** metriğiyle karşılaştırılır. Eşitlikte any character F1,
  o da eşitse config'teki ilk checkpoint seçilir.
- Span sınırları ortak decoder ile üretilir. İki pozitif sözcük arasındaki
  boşluklar span içine köprülenir; F1 bu boşluk karakterlerini de sayar.
  Örtüşen aynı tür anotasyonlar metrikte birleşimdir; farklı türler korunur.
- Model girdileri yalnız kaynak + cevap. Validation gold yalnız skorlamada
  kullanılır; model/önbellek oluşturma yolunda gold etiketleri yoktur.
- Bu seçim validation'a uyarlanmıştır. Validation sonucu bağımsız nihai
  başarı iddiası değildir. Resmî test bu script tarafından kabul edilmez.

## Çalıştırma

Diğer merkezî GPU/CPU job'larının bitmesi beklenir. Repo kökünde:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp/project2_ragtruth' -- env HSA_ENABLE_DXG_DETECTION=1 HF_HUB_CACHE=.cache/hf-hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/banking77-venv/bin/python scripts/run_job.py configs/jobs/transformer_calibration.json
```

Her checkpoint için 2.240 validation cevabında bir inference geçişi yapılır.
Kaynak pencereleri ve örtüşen cevap parçaları mevcut kuralla birleştirilir.
Tür başına dört karakter-olasılık dizisi `probabilities/candidate-N.jsonl.gz`
dosyasına, checkpoint/veri kimliği ve SHA-256 değeri yanında saklanır.
Eşikler bu önbellekten, GPU inference'ı tekrar etmeden karşılaştırılır.

Çıktılar:

- `calibration.json`: seçilmiş checkpoint hash'i, dört eşik, veri kimliği.
- `calibration_report.json`: her eşikteki skorlar, iki checkpoint'in sabit ve
  kalibre edilmiş sonuçları; seçim kuralı/config/code hash'leri.
- `validation_predictions.jsonl`: seçilmiş modelin kalibre edilmiş tahminleri.
- `validation_predictions_candidate_N.jsonl`: adayın seçilmiş eşikleriyle tahmin.
- `validation_predictions_default_N.jsonl`: adayın orijinal sabit eşiğiyle tahmin.
- `probabilities/`: birleştirilmiş karakter olasılıkları, response metin hash'leri
  ve önbellek checksum/kimlik dosyaları. Tamamı Git dışındadır.

Bu doküman kod teslimini açıklar; gerçek kalibrasyonun çalıştırıldığını veya
F1'ın arttığını tek başına iddia etmez. Sonuçlar koşum dosyasından raporlanır.

## Seçilmiş artifact ile tahmin

```powershell
python scripts/transformer.py predict --checkpoint outputs/runs/TRAIN/checkpoints/epoch-2 --calibration outputs/runs/CALIBRATION/calibration.json --split validation --device cuda --output outputs/calibrated-validation.jsonl
```

Checkpoint yolu artifact'te belirtilen seçilmiş adayla aynı olmalıdır.
`checkpoint.json` checksum'ı checkpoint'in config, dosya manifesti ve veri
metadata'sını bağlar; normal model loader ayrıca tüm ağırlık/tokenizer
checksum'larını denetler. Yanlış checkpoint veya eğitim/validation kimliği
reddedilir. Artifact verilmezse önceki sabit eşik davranışı aynen sürer.

Resmî test yalnız mevcut donmuş protokol kapısından çalışabilir. Kalibre
tahminde job girdileri ek olarak `calibration.json`,
`scripts/calibrate_transformer.py` ve onun kullandığı mevcut
`scripts/baseline_crf.py` dosyasını hash'lemelidir. Eşikler resmî testte
ayarlanmaz; bu scriptte `--split test` geçerli değildir.

## Kontroller

`python -m unittest discover -s tests -p test_calibration.py -v`: nadir türü
düşük eşikle bulma, boşluk politikası, eşitlik kuralı, yanlış checkpoint/veri,
bozuk önbellek, donmuş test artifact gereksinimi ve gold'suz CPU CLI inference.
Son test orijinal checkpoint'in değişmediğini ve `--calibration` olmadan
eski davranışın sürdüğünü kontrol eder. Mevcut 26 Transformer testi de geçti.
