# Validation ile tür başına eşik seçimi

> **2 Kasım 2026 inceleme düzeltmesi:** Tarihsel macro-first ve yeni any-first
> seçim ayrı modlardır. Eski config/job dosyaları ve eski sonuç JSON'ları
> byte düzeyinde korunur. Aşağıdaki 1 Kasım bölümleri ara geliştirme kaydıdır;
> güncel mod, precision sınırı ve provenance kuralları belgenin sonundadır.

> Tarihsel kayıt: Bu belge ilk plan/teslim/kontrol sırasındaki durumu anlatır.
> Güncel teknik sonuçlar [FINAL_REPORT.md](../results/FINAL_REPORT.md), nihai
> kararlar [FINAL_PROTOCOL.json](FINAL_PROTOCOL.json) içindedir. Kullanıcının
> son atamasıyla sunum dışındaki teknik tamamlama işi 1. kişi tarafından yapıldı;
> ilk uygulamaların katkı sahipliği korunur.

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

## Düzeltme — 1 Kasım 2026: seçim metriği ve FP koruması

Kod incelemesinde iki sorun doğrulandı: (1) `scripts/calibrate_transformer.py`
checkpoint seçimini `macro_f1_types`'ı birincil anahtar olarak kullanıyordu
(`any` F1 yalnız eşitlik bozucuydu); dört türün ortalaması tek bir türdeki
kayda değer bir `any` F1 düşüşünü maskeleyebilir. (2) `tune_thresholds` hiçbir
yanlış pozitif sınırı olmadan salt F1'e göre seçim yapıyordu: nadir bir türü
neredeyse tüm karakterleri pozitif işaretleyecek kadar düşük bir eşik, "hiç
tahmin etme" (F1 = 0) karşısında ölçülemeyecek kadar küçük ama sıfırdan farklı
bir F1 ile kazanabilir.

Düzeltme (`scripts/calibrate_transformer.py`, bağımsız test edilebilir
`select_checkpoint`/`tune_thresholds` fonksiyonları): checkpoint seçimi artık
önce `any` F1'e, sonra `macro_f1_types`'a bakıyor; eşik araması, sıfır tahmin
dışında precision < `MIN_PRECISION_FLOOR` (0.01 = validation'da en az 100
tahmin edilen karakterden biri gerçek pozitif olmalı) olan eşikleri eler; hiçbir
eşik bu tabanı geçmezse en yüksek (en muhafazakâr) eşik tutulur. Aynı mantık
`scripts/baseline_crf.py`'ye `tune_type_thresholds`/`decode_per_type` olarak
eklendi: tek paylaşılan posterior eşiği yerine her tür kendi karakterlerine
göre bağımsız ayarlanabiliyor (ek çıktı, mevcut `selected` alanını değiştirmez).

`configs/calibration.json` ve `configs/calibration_weighted_policy.json`
**değiştirilmedi**: SHA-256'ları `docs/FINAL_PROTOCOL.json` içinde donmuş
kararın kanıt zinciri olarak sabitlenmiştir. Düzeltilmiş metrik adıyla yeni bir
koşum için `configs/calibration_v2.json` / `configs/calibration_weighted_policy_v2.json`
eklendi. Bu düzeltme yalnız kod/testler seviyesindedir; gerçek checkpoint'ler
bu ortamda yok, dolayısıyla tam veri üzerinde yeniden kalibrasyon çalıştırılmadı.
`scripts/build_technical_report.py` kasıtlı olarak değiştirilmedi: o script
donmuş, tamamlanmış `results/FINAL_REPORT.md` kararının kendisini doğrular; bu
düzeltme mevcut tarihi değil, yalnızca yeni koşumları etkiler.

## Ek düzeltme — 1 Kasım 2026: default/global/tür-başına karşılaştırması

İlk düzeltme checkpoint seçimini düzeltti, ama her checkpoint için nihai
eşik hâlâ koşulsuz olarak tür-başına (`per_type`) sonucu kullanıyordu —
`default_metrics` hesaplanıp kaydediliyordu ama hiçbir zaman gerçekten
karşılaştırılmıyordu. Gerçek CRF ölçümü (`docs/BASELINE_CRF.md`, 1 Kasım 2026)
tür-başına ayarın detection F1'i düşürebileceğini kanıtladı (macro %19.99→%20.84,
any %39.08→%38.84). Aynı riski kapatmak için `tune_global_threshold` (CRF'nin
paylaşılan eşik aramasıyla aynı mantık, tek değer tüm türlere) ve
`select_best_configuration` eklendi: her checkpoint artık default/global/
per_type'ı hesaplayıp detection F1'i (eşitlikte macro F1) en yüksek olanı
seçiyor. `calibration_report.json`'da her üç yapılandırma da (`configuration_metrics`)
ve kazanan (`configuration_winner`) ayrıca kaydedilir.

## İnceleme düzeltmesi — 2 Kasım 2026: geriye uyumlu açık modlar

`scripts/calibrate_transformer.py` mode'u açık config'in `selection_metric`
alanından çıkarır. İstenirse `--selection-mode legacy_macro` veya
`--selection-mode successor_any` ile ayrıca sabitlenir; flag ile config
çelişirse herhangi bir inference/çıktı oluşturmadan reddedilir.

| Mod | Config metriği | Decoder/checkpoint seçimi |
|---|---|---|
| `legacy_macro` | `macro_f1_types` | Orijinal bağımsız tür eşikleri, precision tabanı yok; macro F1, ardından any F1, ardından erken aday. Default yalnız tanı karşılaştırması. |
| `successor_any` | `detection_f1_then_macro_f1_types` | Default/global/per_type/joint içindeki uygun adaylar; önce any F1, sonra macro F1, sonra ilan edilmiş sıra. Checkpoint eşitliğinde erken aday. |

Eski `configs/calibration.json`, `configs/calibration_weighted_policy.json`
ve `configs/jobs/transformer_calibration.json` değiştirilmedi. Eski job
default macro config'iyle tekrar uyumludur. Bu, geçmiş inference'ın donmuş
kod hash'iyle aynı byte'lara sahip olduğunu iddia etmez: tam tarihsel replay
ilgili eski commit/ortamı gerektirir. Mevcut resmî test yeniden ayar seçimi
için kullanılmaz.

`complete_technical_pipeline.py` tarihsel weighted/unweighted aile kararını
ve config'ini **macro-first** tutar; oluşturduğu calibration job'ı açık
`legacy_macro` flag'i taşır. Yeni any-first aile deneyi bu tarihsel pipeline'a
sessizce geçirilmez. Saf `choose_experiment` fonksiyonuna successor metriği
ancak açıkça verilebilir; mode'ları farklı decoder artifact'leri karıştırılmaz.

Successor calibration ayrı komut/job üzerinden kullanılır:

```text
python scripts/calibrate_transformer.py --config configs/calibration_v2.json --selection-mode successor_any --output outputs/new-successor-calibration --device cuda
python scripts/run_job.py configs/jobs/transformer_calibration_v2.json
```

Buradaki `v2` **calibration politika sürümüdür**. Ayrı `.cache/ragtruth-v2`
worktree'sindeki fresh internal-holdout çalışmasıyla aynı deney değildir.
Yeni config'teki checkpoint yolları kullanılacak gerçek dev çalışmasına ait
olmalıdır. Bu inceleme sırasında komutlar gerçek veri/model üzerinde çalıştırılmadı.

## Precision tabanı: uygun aday yoksa reddetme

Successor modunda dört raw türün her biri için validation precision en az
0,01 olmalı veya o tür hiç karakter tahmin etmemelidir. Bu yalnız bir dev
uygunluk koşuludur; başka veride aynı precision'ın sağlanacağını garanti etmez.
Default ve global adaylar da aynı koşuldan geçer. Özellikle any F1 iyi olsa
bile yanlış türdeki yoğun işaretleme adayın uygun sayılmasına yetmez.

Hiçbir grid eşiği uygun değilse `max(grid)` değeri otomatik seçilmez;
`NoEligibleThreshold` ile o arama reddedilir. Diğer uygun yapılandırma veya
checkpoint varsa seçim onlar arasında sürer. Hiçbiri uygun değilse işlem
başarısız biter; `calibration.json` ve nihai prediction artifact'i üretilmez.
`calibration_report.json` içinde default metrikleri, tür precision/sayıları ve
reddedilme nedenleri; ayrı default prediction dosyaları tanı için korunur.
Bir checkpoint reddedilip diğeri seçilebildiğinde yeni successor job yine
tamamlanabilir; reddedilen checkpoint için `validation_predictions_candidate_N`
dosyası zorunlu çıktı değildir. Default tanı tahminleri her aday için korunur.
Boş validation reddedilir. Gold pozitif olmayan türlerde boş prediction uygundur;
yanlış pozitifleri sırf daha yüksek eşik kalmadığı için kabul etmez.

Legacy macro modu bu yeni precision politikasını uygulamaz; geçmiş kararın
aynı seçim kuralını korumak için taban açıkça sıfırdır. Bu legacy modun
yanlış alarmları engellediği anlamına gelmez.

## Joint decoder artifact'i, üretici ve süre kayıtları

Yeni `calibration.json` aynı dört `thresholds_by_type` alanını ve checkpoint/veri
bağını kullanır; mevcut inference loader ile yüklenir. Ek olarak
`selected_configuration`, sürümlü `selection_policy` ve gerçek üretici
`producer` bilgileri bulunur. Joint kazanırsa yalnız bir karşılaştırma JSON'ı
değil, doğrudan kullanılabilir normal decoder artifact'i dışa aktarılır.

Producer kaydı script ile kullanılan hizalama/model/CRF kaynaklarının gerçek
SHA-256'larını, varsa Git commit'ini ve dirty durumunu kaydeder. Kod veya config
koşum sırasında değişirse artifact seçimi reddedilir. `calibration_report.json`
başlangıç/bitiş UTC zamanlarını, calibration'ın toplam elapsed süresini ve
ayrı threshold arama sürelerini kaydeder. Cache metadata'sındaki inference
süresi ile CPU eşik aramasının süresi aynı iş değildir.

`results/session_2026-11-01/` eski oturumunun JSON'ları değiştirilmedi.
Oradaki kalibrasyon raporunun producer hash'i mevcut committed kaynakla
eşleşmez ve joint denemesi ayrıca cache üstünde yapılmıştır. Eski global
decoder artifact'ini sonradan joint sonucuymuş gibi yeniden adlandırmak veya
eski metrikleri yeni producer hash'iyle üretildi diye sunmak doğru değildir.
Yeni producer/timing alanları yalnız bu kodla gerçekten yapılacak yeni koşumlara
aittir; arşivdeki eksik ağırlık/prediction dosyalarını veya provenance bağını
geriye dönük olarak oluşturmaz.
