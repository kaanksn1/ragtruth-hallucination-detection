# CRF baseline — 3. kişi

> Tarihsel kayıt: Bu belge ilk plan/teslim/kontrol sırasındaki durumu anlatır.
> Güncel teknik sonuçlar [FINAL_REPORT.md](../results/FINAL_REPORT.md), nihai
> kararlar [FINAL_PROTOCOL.json](FINAL_PROTOCOL.json) içindedir. Kullanıcının
> son atamasıyla sunum dışındaki teknik tamamlama işi 1. kişi tarafından yapıldı;
> ilk uygulamaların katkı sahipliği korunur.

Durum: **kod, testler ve koşum config'leri hazır; tam eğitim 1. kişinin PC'sinde
bekliyor.** Bu belgede tam veri sonucu yoktur. Sonuçlar, hata analizi ve slayt
notları merkezî koşum döndükten sonra eklenecek. Resmî test kullanılmadı.

## Ne yapıyor?

`kaynak (source_text) + cevap (response) → cevaptaki her kelime için O/B/I → karakter aralıkları`

Model girdisi yalnız `schema.MODEL_INPUT_FIELDS = (source_text, response)`.
`id`, `source_id`, `task_type`, `generator`, `quality` ve anotasyon alanları
(`meta`, `implicit_true`, `due_to_null`, gold `text`) özelliklere girmez.
Testler, etiket ve metadata değiştiğinde özelliklerin değişmediğini doğrular.

**Mimari ve somut model.** Mimari: linear-chain CRF (koşullu rastgele alan).
Somut model: CRFsuite (`python-crfsuite==0.9.12`), L-BFGS ile eğitim, L1 (`c1`)
ve L2 (`c2`) düzenlileştirme, el ile yazılmış ikili özellikler.

### Neden tür başına ayrı CRF?

Ortak veri sözleşmesi her tür için ayrı O/B/I dizisi verir (`bio_by_type`).
Dört türden her biri için bağımsız bir CRF eğitilir. Farklı türdeki örtüşen
anotasyonlar (train'de 23 çift) böylece kaybolmaz; tek bir 9 etiketli BIO dizisi
aynı kelimeye iki tür veremezdi. Bedeli: türler arası bağımlılık modellenmez ve
eğitim 4 kat sürer. Tahmin dosyasında dört türün aralıkları birlikte yazılır.

### CRF nasıl karar veriyor?

Bir cevap için etiket dizisi `y = (y_1 … y_n)` skoru:

`score(x, y) = Σ_i Σ_k w_k · f_k(x, i, y_i)  +  Σ_i T[y_(i-1), y_i]`

- `f_k(x, i, y_i)`: "i. kelime `src:content_new` ve etiketi B" gibi gözlem özellikleri (unary).
- `T[a, b]`: etiket geçiş ağırlığı (ör. `B→I` yüksek, `O→I` düşük).
- `P(y | x) = exp(score(x, y)) / Z(x)`; `Z(x)` bütün diziler üzerinden normalizasyondur
  (forward algoritması). Eğitim, train log-likelihood'unu `c1·|w|₁ + c2·‖w‖²` cezasıyla
  en büyütür.

**Çözümleme (decoding).** İki yöntem raporlanır ve validation ile seçilir:

1. **Viterbi:** en yüksek skorlu tek diziyi seçer (dinamik programlama).
2. **Posterior eşik:** forward-backward ile her kelime için `P(y_i ≠ O | x)`
   hesaplanır; eşik `τ` üstündeki ardışık kelimeler bir aralık olur.

Neden ikincisi? Train'de kelimelerin büyük çoğunluğu `O`
(ör. Evident Baseless Info'da yaklaşık %97). En olası *dizi* çoğu zaman
"hepsi O" çıkar; eşik, precision/recall dengesini açıkça ayarlamayı sağlar.
Eşik yalnız validation ile seçilir.

**BIO kısıtı.** Sert kısıt uygulanmaz; `O→I` gibi geçişlerden öğrenilmiş geçiş
ağırlıkları kaçınır. Viterbi çıktısında önünde B/I olmayan `I` sayısı raporlanır
(`invalid_i_starts`) ve `tags_to_char_spans` böyle bir `I`'yı yeni aralık başlangıcı
olarak okur. Posterior çözümlemede her aralık kurgu gereği B ile başlar.

## Özellikler

Cevap `alignment.word_tokenize` ile kelimelere ayrılır (2. kişinin CRF dosyasıyla aynı).

| Grup | Özellikler |
| --- | --- |
| Kelime | küçük harf kelime, son 3 harf, kelime şekli (`Xxx`, `dd`, `XX-dd`), büyük harf/sayı/noktalama |
| Bağlam | önceki/sonraki kelime ve şekli, cümle başı, cümle sırası (0, 1, 2, 3+) |
| **Kaynak** | kelime kaynakta var mı; içerik kelimesi kaynakta yeni mi; sayı kaynakta yeni mi; cümle ortasında kaynakta olmayan büyük harfli kelime; önceki/sonraki bigram kaynakta var mı; komşu kelime yeni mi; ±3 penceredeki yeni içerik kelimesi sayısı |
| **Kaynak, cümle** | cümledeki içerik kelimelerinin kaynakta olmayan oranı (0 / ≤%25 / ≤%50 / >%50); cümlede kaynakta olmayan sayı var mı |

"İçerik kelimesi": sayı içeren veya küçük bir İngilizce stopword listesinde
olmayan kelime. Kaynak karşılaştırması büyük/küçük harfe duyarsız tam kelime
eşleşmesidir; eş anlam, çekim eki veya biçim farkı (`2,000` ↔ `2000`) yakalanmaz.
Kaynak metni, üretici modele verilen prompt'un tamamıdır (talimat + soru + içerik).

Eğitimde 2'den az görülen özellikler atılır (`feature.minfreq = 2`). Bütün
sözlük ve ağırlıklar yalnız train'den öğrenilir; validation/test üzerinde fit yok.

`features.source_features` ayarı ana ızgarada gerçekten uygulanır; `false`
seçilirse ana varyant yalnız cevabı kullanır ve ikinci bir aynı kontrol eğitilmez.
Varsayılan `true`dur. `quality_filter=good` ve `algorithm=lbfgs` bu teslimin
desteklenen koşullarıdır; geçersiz ayarlar sessizce yok sayılmaz.

## Deney tasarımı

| Ayar | Değerler |
| --- | --- |
| `c2` (L2) | 0.01, 0.1, 1.0 (`c1 = 0.1`) |
| Çözümleme | Viterbi; posterior `τ` ∈ {0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05} |
| Sabit | L-BFGS, `max_iterations = 150`, `possible_transitions = true`, `minfreq = 2` |
| Veri | `quality == "good"`: 12.702 train, 2.240 validation cevabı (`data.load_split` varsayılanı) |
| Seçim | validation karakter F1, bütün türlerin birleşimi (`any`); eşitlikte ilk satır |
| Kontrol | seçilen ayar ve çözümleme ile **yalnız cevap** CRF'i (kaynak özellikleri kapalı) |

Seçim metriği karakter düzeyinde micro precision/recall/F1'dir: her cevapta gold
ve tahmin aralıklarının birleşimindeki karakterler karşılaştırılır. Tür bazında
aynı hesap ve dört türün macro F1'i de raporlanır. **Bu metrik yalnız CRF ayar
seçimi içindir; modellerin ortak karşılaştırması 5. kişinin scorer'ıyla yapılır.**
`implicit_true` ve `due_to_null` anotasyonları ortak veri sözleşmesindeki gibi hedefte kalır.

**Yalnız cevap kontrolü.** Aynı mimari, aynı veri, aynı ayar ve çözümleme; tek fark
kaynak özellikleri. Böylece CRF içinde kaynak bağlamının katkısı ölçülür. CRF ile
Transformer arasındaki fark ise hem mimari hem girdi değiştiği için yalnız bağlama
atfedilmemeli.

## Çalıştırma

```powershell
python -m venv .venv
# Windows: .venv/Scripts/python.exe; diğer sistemler: .venv/bin/python
# Aşağıdaki python komutlarını bu proje ortamı etkinleştirilmişken çalıştırın.
python -m pip install -r requirements-crf.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/run_job.py configs/jobs/crf_smoke.json        # ~1 dk, 300/100 cevap
python scripts/run_job.py configs/jobs/crf_validation.json   # tam eğitim + validation
```

`configs/crf.json` içindeki `split_summary` dosyasının dört işlenmiş veri hash'i
eğitimden önce doğrulanır. Uyuşmazlık eğitimi durdurur. Sentetik testler kendi
veri özeti ve hash'lerini açıkça sağlar; ortak veri kimliği kontrolü atlanmaz.
Yanıt kimliklerinin tekrarı, eksik CRF satırı, bozuk token/offset eşlemesi ve
train/validation arasında ortak cevap kimliği reddedilir.

Çıktılar `outputs/runs/<zaman>-crf-validation/` altında:

| Dosya | İçerik |
| --- | --- |
| `crf_report.md` | Izgara tablosu, seçim, yalnız-cevap kontrolü, eğitim süreleri |
| `crf_summary.json` | Bütün skorlar, veri hash'leri, ortam, en güçlü özellik ve geçiş ağırlıkları |
| `validation_predictions.jsonl` | Seçilen model, ortak tahmin biçimi (`{"id", "spans"}`) |
| `validation_predictions_response_only.jsonl` | Yalnız cevap kontrolü, aynı biçim |
| `models/source/`, `models/response_only/` | 4 `.crfsuite` modeli + `model_config.json` (ayar ve çözümleme) |

Kaydedilen model config'i etiket şemasını, veri kimliğini ve dört modelin
SHA-256'sını içerir. Tahminden önce bunlar doğrulanır. Mevcut eğitim
checkpoint/sonuçlarının veya tahmin dosyasının üzerine sessizce yazılmaz.
Job, model dosyalarını da beklenen çıktılara ekleyip hash'lerini kaydeder.

Windows'ta `Masaüstü` gibi ASCII dışı yollar CRFsuite'in native dosya API'sinde
sorun çıkarabilir. Eğitim dosyası ASCII adlı geçici klasörde oluşturulup Python
ile gerçek hedefine kopyalanır; yükleme bellekteki model baytlarıyla yapılır.
Modelin gerçekten yazıldığı kontrol edilir. Geçici klasörün kendisi de ASCII
olmalı; mevcut makinenin standart Windows temp klasörü bu koşulu sağlar.

Kısa koşumun validation tahminlerini aynı alt kümede yeniden üretmek için:

```powershell
python scripts/baseline_crf.py predict --model-dir outputs/runs/<smoke-koşumu>/models/source --split validation --limit 100 --output outputs/crf_smoke_reproduced.jsonl
```

`--limit` yalnız kısa validation tekrar üretimi içindir; resmî testte kabul
edilmez. Tahmin yükleyicisi gold aralıklarına veya BIO hedeflerine ihtiyaç duymaz.

Protokol dondurulduktan sonra test tahmini yeniden eğitim yapmadan alınır
(`prepare_data.py --include-test` ile test dosyaları üretildikten sonra):

Resmî test için doğrudan `baseline_crf.py predict --split test` çağrısı
engellenir. [RUNNING.md](RUNNING.md) sözleşmesine göre `stage=predict`,
`split=test` ve donmuş protokollü bir job hazırlanır. Job'ın `inputs` listesi,
testin iki işlenmiş dosyasını, CRF/koşum scriptlerini, ortak veri/hizalama/şema
modüllerini, bağımlılık listesini, model config'ini ve dört model dosyasını içerir.
Mevcut modelin tam yolu ve gerçek komut argv'si bu config'te
sabitlenir. Örnek aşağıdaki dosya **protokol tamamlanınca oluşturulacak** dosyadır:

```powershell
python scripts/run_job.py configs/jobs/crf_test.json
```

`run_job` doğrulamadan sonra kendi koşum kaydının yolunu çocuk sürece verir.
CRF kodu donmuş protokolü ve aynı komut/girdilerle çalıştığını tekrar kontrol
eder. Gerçek resmî test koşumu yapılmadı; boş bir test config'i teslim edilmedi.

**Süre ve bellek (yerel ölçüm, Apple Silicon, tek çekirdek).** 3.000 train
cevabında bir tür için özellik ekleme ~6 sn, 150 iterasyonluk eğitim ~15 sn sürdü.
Tam veri (~4,2 kat) için tahmin: ızgara ~15 dk, yalnız-cevap kontrolü ~6 dk, toplam
~25 dk. Aynı anda tek CRF verisi bellekte tutulur (tür tür eğitim). Job süre sınırı 3 saat.

## Geliştirme notları (sonuç değildir)

Tasarım, validation'ın ilk 200–500 cevabıyla yapılan küçük yerel denemelerle seçildi.
Bu sayılar alt kümeye aittir ve raporlanacak sonuç değildir.

- 500 train cevabıyla ilk sürüm eğitimi ezberledi (train karakter F1 0.997,
  validation 0.005). En güçlü ağırlıklar `w=kidnappings` gibi tekil kelimelerdi.
  Değişiklikler: ±2 uzaklıktaki kelime kimlikleri kaldırıldı, `minfreq = 2`,
  cümle düzeyinde kaynak yeniliği özellikleri eklendi.
- 3.000 train / 500 validation cevabında, `c1 = c2 = 0.1` ile validation F1 (any):
  Viterbi 0.090; posterior `τ = 0.1` 0.146. Yalnız cevap aynı koşulda 0.032 / 0.061.
  Conflict türlerinde F1 sıfıra yakındı: kelime örtüşmesi, kaynakta geçen
  kelimelerle kurulmuş yanlış bir ifadeyi (ör. yer değiştirmiş varlıklar) görmez.

Bu gözlemler tam eğitimde tekrar kontrol edilecek; tam sonuçlar gelmeden genelleme yapılmamalı.

## Bekleyen işler (3. kişi)

1. Merkezî tam koşumun `crf_report.md` ve `crf_summary.json` çıktısını yorumlamak.
2. Validation'dan en az üç hata örneği (kaçırılan conflict, yanlış işaretlenen
   kaynakta olmayan ama doğru bilgi, aralık sınırı hatası) ve ağırlık açıklaması.
3. 1–2 slayt ve konuşmacı notu.
4. Bütçe varsa: 4. kişinin Transformer kodu için yalnız-cevap deney config'i.

## Entegrasyon incelemesi — 31 Ekim 2026

1. kişi, kullanıcı atamasıyla veri kimliğinin zorunlu kontrolünü, model
checksum'larını, kayıtsız test çağrısının engellenmesini, kaynak ayarının
uygulanmasını ve sınır durumlarının kontrollerini ekledi. CRF modeli ve ilk
deney tasarımı 3. kişinin teslimidir; bu düzeltmeler kişisel sonuç analizi veya
slaytlarını onun adına tamamlamaz. CI, küçük gerçek CRFsuite train/predict
kontrolünü çalıştırmak için sabit CRF paketini kurar.

### Merkezî kısa koşumun doğrulanması

- 1. kişinin Windows PC'si, Python 3.12.0 ve ayrı proje `.venv` ortamı;
  `python-crfsuite==0.9.12`. GPU/önceki proje ortamı değiştirilmedi.
- Temiz `4b200c2` commit'iyle `configs/jobs/crf_smoke.json`: 300 train / 100
  validation, 3 ayar × 4 tür + seçilen ayarda yalnız-cevap kontrolü.
- Koşum: `outputs/runs/20261030T212347530569Z-crf-smoke/`, exit code 0,
  wrapper süresi 31,6 saniye. Bu süre tam veri eğitim tahmini değildir.
- Hem kaynaklı hem yalnız-cevap checkpoint'leri yeniden yüklenip aynı 100
  validation cevabında tahmin verdi; iki dosya da ilk çıktısıyla bayt bayt aynı.
  Makine kaydı: yerel `outputs/crf_review_validation.json`.
- 75 test senaryosu: ayrı CRF ortamında 71 geçti / HF-PyTorch gerektiren 4
  test atlandı; mevcut HF ortamında 73 geçti / CRF gerektiren 2 test atlandı.
  Koşullu atlananların her biri diğer ortamda çalıştırıldı. Native Windows
  yol kontrolü gerçek Türkçe klasör adıyla da geçti.
- Önceki başarısız Windows yol denemesi ayrı koşum logunda korundu; üzerine
  yazılmadı. Tam veri sonucu, hata analizi, slayt ve resmî test hâlâ bekliyor.

## Tam veri koşumu, joint eşik araması ve nadir tür incelemesi — 1 Kasım 2026

Gerçek veri indirildi/hazırlandı (dört işlenmiş dosya, donmuş hash'lerle bayt
bayt eşleşti) ve `configs/crf.json` ile tam koşum yapıldı: `outputs/crf_local_validation_run1/`
(run_job sarmalayıcısı değil, git deposu olmadığından `baseline_crf.py` doğrudan
çağrıldı; 1497 saniye ≈ 25 dk). Seçilen ayar/çözümleme (`c1_0.1_c2_1`,
`posterior_0.15`, any F1 0.3908) donmuş `results/training_crf.json` ile aynı;
kaydedilen `.crfsuite` dosyalarının SHA-256'sı da donmuş `FINAL_PROTOCOL.json`
ile bayt bayt eşleşti (seed 42, deterministik L-BFGS).

**Yeni: birlikte (joint) eşik araması** (`tune_joint_thresholds`,
`select_best_configuration`). Tür başına bağımsız arama (`tune_type_thresholds`)
her türün kendi F1'ini en iyiler; bu, birleşik 'any' F1'i kazara düşürebilir.
Gerçek validation'da ölçülen: paylaşılan eşik any F1 **39.08%**, bağımsız tür
başına **38.84%** (düşüş), joint arama **39.22%** (artış, +0.14 puan; bu önceki
iki yapılandırmayı da geride bırakır). Dört yapılandırma arasından detection
F1'i en yüksek olanı seçen `select_best_configuration` bu koşumda **joint**'i
seçti. Hiçbir yapılandırmada herhangi bir tür 10 katı tahmin/gold oranını
geçmedi (en yüksek: tür başına Subtle Baseless Info 3.26x).

**Subtle Conflict: eşik sorunu değil.** Zaten eğitilmiş modelle (yeniden eğitim
yok) gerçek 155 validation gold token'ının üzerindeki en yüksek P(B/I)
olasılığı **0.0036**'dır (ortalama 0.0004); yapılandırılmış ızgaradaki eşikler
bu örnekleri yakalayamıyor. Bu, bütün olası eşikler veya CRF modelleri için
imkânsızlık kanıtı değildir.
0.001 gibi aşırı düşük bir eşikte (yapılandırılmış ızgaranın altında, hiçbir
seçim mantığı tarafından önerilmez) FP patlaması gerçekten gözlenir: 830 gold
karaktere karşı 136.693 tahmin (**164.69x**, F1 %0.07) — projenin orijinal
aşırı FP kaygısının somut mekanizması budur, ama mevcut `[0.05,0.5]` ızgarası
ve %1 precision tabanı bunu zaten dışarıda tutuyor.

**Hedefli regularizasyon denemesi** (yalnız Subtle Conflict tipi, ~225 sn,
yeniden eğitim gerekti): mevcut ayar (`c1=0.1,c2=1.0,minfreq=2`) train
gold token'larında gerçek sinyal öğreniyor (ortalama P=0.232, maks 0.917) ama
validation'a hiç taşımıyor (ortalama 0.0004) — klasik ezberleme, `top_weights`
içindeki aşırı özgül kelimelerle (`feminist`, `vegetarian`, `misleading`)
tutarlı. Çok daha az regularizasyon denemesi (`c1=0,c2=0.001,minfreq=1`,
88.347 özellik) **daha da kötü** sonuç verdi (train'de bile ortalama P=0.0004):
aynı 150 iterasyon sınırıyla yakınsama doğrulanmadı. Bu tek deneme,
regularizasyonun bütünüyle elendiğini göstermez; özellikler, az veri ve
optimizasyon bütçesinin etkileri birbirinden ayrılmadı.

**Sonuç:** Bu koşum/özellik kümesi Subtle Conflict'i güvenilir biçimde
öğrenemedi. Bütün CRF mimarileri için bir başarı tavanı kanıtlanmadı. Eski V1
dev'de Transformer bu tür için küçük, sıfır olmayan F1 üretmişti; yeni oturumun
joint Transformer raporunda ve eski resmî testte bu türün F1'ı sıfırdır.

## Seçilen decoder'ın yeniden yüklenmesi — 2 Kasım 2026

Yeni eğitimler dört yapılandırmanın kazananını ayrıca
`models/source_selected/model_config.json` içinde kaydeder. Tür başına eşikler
`posterior_by_type` stratejisiyle standart `predict` komutunca uygulanır.
Eşleşen dev çıktısı `validation_predictions_selected.jsonl`; yollar
`crf_summary.json` içindeki `best_configuration` alanında belirtilir.

```text
python scripts/baseline_crf.py predict --model-dir RUN_DIR/models/source_selected --split validation --output outputs/NEW_SELECTED.jsonl
```

Tarihsel global `models/source` ve `validation_predictions.jsonl` dosyalarının
anlamı korunur. Yeni export, eskiden eğitilmiş dosyaları veya resmî test
sonuçlarını değiştirmez; geçmiş session raporundan yeni bir model uydurulmaz.
Yalnız-cevap kontrolü eski global decoder'ı kullanmaya devam eder; onu yeni
selected kaynaklı modelle tek-değişkenli ablation gibi kıyaslamayın.

Per-type/joint aramada hiçbir grid eşiği precision koşulunu sağlamıyorsa
artık hata verilir; en yüksek eşik otomatik olarak güvenli kabul edilmez.
%1 precision tabanı yalnız bu aday aramalarının kısıtıdır; default/global
diagnostic adaylarına veya bütün gelecekteki verilere yanlış-alarm garantisi değildir.
