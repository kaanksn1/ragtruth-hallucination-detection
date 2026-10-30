# Kaynak bağlamlı çok etiketli Transformer — 4. kişi

Kod teslimi: 31 Ekim 2026. Tam eğitim, gerçek pretrained modelin GPU pilotu ve
başarı ölçümü henüz yapılmadı. Tam train/predict koşumları 1. kişinin PC'sinde
çalıştırılır. Bu belge deney sonucunu veya donmuş test protokolünü temsil etmez.

## Dosya teslimi

| Dosya | Değişiklik |
|---|---|
| `transformer.py` | Yeni: model, maskeli loss, streaming Dataset/DataLoader, decoder, checkpoint |
| `scripts/transformer.py` | Yeni: train/predict CLI, veri doğrulama, kayıt, test protokol kapısı |
| `scripts/smoke_transformer.py` | Yeni: yerel pretrained ağırlıklarla iki sentetik cevap/tek CPU optimizer adımı |
| `configs/transformer.json` | Yeni: model/seed/pencere/eğitim ayarları |
| `configs/jobs/transformer_train.json` | Yeni: 1. kişinin merkezî GPU koşumu |
| `requirements-transformer.txt` | Yeni: mevcut CRF requirements dosyasından bağımsız bağımlılıklar |
| `tests/test_transformer.py` | Yeni: 20 işlev/entegrasyon testi |
| `.github/workflows/transformer.yml` | Yeni: ağırlık indirmeyen CPU test workflow'u |
| `docs/TRANSFORMER.md`, `docs/TRANSFORMER_PR.md` | Yeni: bu rehber ve PR taslağı |
| `docs/TRANSFORMER_REVIEW.md` | Yeni: PR öncesi inceleme, düzeltmeler ve gerçek CPU test kanıtı |
| `README.md` | Yalnız Transformer rehber bağlantısı ve kısa durum paragrafı eklendi |

Yerel `artifacts/pr_review_all_tests.log` test günlüğü ve
`artifacts/pretrained-cpu-smoke/report.json` gerçek ağırlık kontrolünün kaydıdır;
Git'e alınmaz. `.cache` model/tokenizer için, `artifacts` test checkpoint'i için
kullanılır; bu dosyalar mevcut `.gitignore` tarafından dışlanır.

## İnceleme ve entegrasyon

README, AGENTS, TEAM, DATA_FORMAT, DATA_WINDOWING, PROTOCOL_DRAFT, ortak veri
kodları, CRF ve koşum/test arayüzleri incelendi. Mevcut veri/CRF/koşum kodları
değiştirilmedi. `ragtruth_data.features.build_features`, `pad_batch`,
`aggregate_window_scores` ve `char_scores_to_spans` doğrudan kullanılıyor.

- Girdiler yalnız `source_text` (üretim promptu, soru ve kaynak) ile orijinal
  `response`. Metadata, kaynak/cevap kimlikleri ve gold açıklamalar modele girmez.
- `data/processed/{train,validation}.jsonl`, ortak summary SHA-256 değerleri
  ve `split_manifest_v1.csv` doğrulanır. `quality=good` filtresi korunur.
- Train/validation kaynak ve içerik grupları yeniden bölünmez; ID, source_id
  ve group_id çakışmaları reddedilir. Resmî test eğitim sırasında okunmaz.
- Mevcut CRF dört ayrı BIO etiketleyicisi kullanıyor. Transformer'da dört
  bağımsız binary çıktı var; BIO sınıf sayısı veya softmax kullanılmıyor.
- Ortak `evaluate.py` incelenen main'de yok. Mevcut CRF'nin `char_scores` fonksiyonuyla
  sentetik çıktı uyumu test edildi; bu, 5. kişinin henüz teslim edilmemiş
  ortak scorer'ının doğrulandığı anlamına gelmez.

## Mimari ve etiket sırası

`AutoModel` encoder → dropout → `Linear(hidden_size, 4)` → dört logit.
Tahminde her logite ayrı sigmoid uygulanır; toplamlarının 1 olması gerekmez.
Encoder'ın tamamı fine-tune edilir. Kaynak tokenları attention'a katılır;
kaynağı değiştiren bir test cevap logits'lerinin değiştiğini doğrular.

Başlangıç encoder/tokenizer: `answerdotai/ModernBERT-base`.
İkisi için de revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` sabittir.
[Model kartı](https://huggingface.co/answerdotai/ModernBERT-base) ModernBERT'in
Transformers 4.48+ desteğini belirtir. Bu teslim 4.57.1 ile test edildi;
[4.57.1 ModernBERT API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/modernbert)
kullanılır. Revision [Hub API](https://huggingface.co/api/models/answerdotai/ModernBERT-base)
üzerinden doğrulandı. FlashAttention kurulumu gerekmez; varsayılan SDPA,
`reference_compile=False` ve isteğe bağlı gradient checkpointing kullanılır.

Sıra doğrudan `ragtruth_data.schema.LABEL_TYPES` sabitinden alınır:

1. `Evident Conflict`
2. `Subtle Conflict`
3. `Evident Baseless Info`
4. `Subtle Baseless Info`

Bu adlar ortak şema, veri raporu ve repodaki RAGTruth örnek anotasyonlarıyla
uyumludur. Örneğin bir tokenın `[1, 0, 1, 0]` hedefi iki farklı türü korur.
Checkpoint'te sıra saklanır; farklı sırayla yükleme reddedilir.

## Tokenizer ve pencereleme

Ortak sözleşmeye uygun giriş sırası **cevap + kaynak penceresi**dir.
`sequence_ids()` ile cevap tokenları seçilir; segment embedding desteği
olmayan ModernBERT'e `token_type_ids` geçirilmez. Fast tokenizer zorunludur.
Orijinal cevap normalize edilmez; tüm span'ler cevap içindeki `[start, end)`
karakter koordinatlarını kullanır.

Varsayılanlar `configs/transformer.json` içinde:

| Ayar | Değer |
|---|---|
| max_length | 2048 |
| context_stride | 128; ortak kod en fazla yarım pencereye sınırlar |
| min_context_tokens | 128 |
| response_overlap | 64 |
| Eğitim pencereleri | `best`: cevap parçası başına metinsel örtüşmesi en yüksek kaynak penceresi |
| Validation/tahmin | bütün pencereler |
| Kaynak pencerelerinin birleştirilmesi | tür başına `min` |
| Örtüşen cevap parçaları | ortak kodun karakter bazlı ortalaması |
| Sigmoid eşiği | sabit 0.5; testte ayarlanmaz |

Uzun cevap örtüşen parçalara bölünür; uzun kaynağın bütün pencereleri tahminde
görülür. Pencere seçimi gold etikete bakmaz. `best` eğitimde bütün kaynağı
görmeyi garanti etmez; kelime örtüşmesi gerçek kanıtın yerini yanlış seçebilir.
`min`, herhangi bir penceredeki düşük skoru korur: destek bulunduğunda yanlış
pozitifi azaltabilir, ilgisiz pencerelerdeki düşük skorlar da yanlış negatif
üretebilir. Bunlar DATA_WINDOWING önerileriyle uyumludur; başarı garantisi
değildir. GPU pilotundan sonra 1024'e geçilirse maliyet/kapsama yeniden ölçülür.

Dataset her seferinde tek cevabın özelliklerini üretir; tüm korpusun padded
token/etiket tensörleri RAM'de tutulmaz. DataLoader batch size pencere sayısıdır,
cevap sayısı değildir. Num_workers=0, epoch başına seed'li cevap sırası kullanılır.

## Loss ve eğitim seçimi

`BCEWithLogitsLoss` yalnız `response_mask & attention_mask` konumları seçildikten
sonra çalışır. Dört sınıf ve geçerli cevap tokenları üzerinde ortalama alınır.
Kaynak/özel/padding tokenları dışlanır; cevap içindeki `[0,0,0,0]` hedefleri
geçerli negatiflerdir. Etiketler float32, şekil `[batch, sequence, 4]`.

Boş batch veya geçerli cevap tokenı olmayan batch differentiable sıfır loss
verir. Eğitim döngüsü bu batch'te optimizer adımını atlar; AdamW weight decay
ve momentum'un parametreleri değiştirmesi engellenir. Bir bölümün tamamında
geçerli cevap yoksa eğitim açık hata verir.

AdamW, gradient clipping ve cevap-token sayısıyla ağırlıklandırılmış gradient
accumulation kullanılır. Son eksik accumulation grubu da doğru normalize edilir.
Varsayılan: batch 1, accumulation 8, lr 2e-5, 3 epoch, seed 42, FP32.
`precision=bf16` yalnız destekleyen CUDA/ROCm cihazlarında açılabilir.
Donanımlar arasında bit düzeyinde aynı sonuç garanti edilmez.

Checkpoint seçimi minimum **validation masked BCE** ile yapılır; eşitlikte
ilk epoch korunur. Validation bütün pencereleri kullandığından uzun örnekler
birden çok katkı yapar. Bu metrik cevap başına birleştirilmiş character F1
değildir. Eşik şimdilik 0.5'tir; sınırlı validation eşik araması ve nihai
F1 tabanlı seçim protokolü 5. kişiyle netleştirilecek. Test etiketleriyle seçim yok.

## Kurulum ve küçük test

Komutlar `AGENTS.md` bulunan gerçek proje kökünden çalıştırılır. ZIP iç içe
açılmışsa dış klasörün bir altındaki `ragtruth-hallucination-detection-main`
klasörüne girin. Python 3.12 önerilir.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-transformer.txt -r requirements-crf.txt
python -m unittest discover -s tests -p test_transformer.py -v
python -m unittest discover -s tests -v
python -m pip check
```

Bu bilgisayarda Windows Store `python` kısayolu çalışmadığı için başlangıçta
`C:\Users\Adnan\AppData\Local\Python\bin\python3.12.exe -m venv .venv`
kullanıldı; sonra `.venv\Scripts\python.exe` ile testler çalıştı.

Ana küçük testler ağ veya model ağırlığı istemez. Küçük, rastgele başlatılmış
iki katmanlı ModernBERT ve sentetik tokenizer/veri kullanır. CLI entegrasyon
testi iki sentetik train ve iki validation cevabıyla bir optimizer adımı yapar.

Opsiyonel gerçek tokenizer kontrolü için yalnız dört JSON indirilir:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('answerdotai/ModernBERT-base', revision='8949b909ec900327062f0ebf497f51aef5e6f0c8', allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'], local_dir='.cache/modernbert-tokenizer')"
python -m unittest discover -s tests -p test_transformer.py -v
```

Bu kontrol gerçek tokenizer'ı küçük rastgele encoder ile çalıştırır ve sekiz
committed örnekte hizalamayı sınar. Büyük pretrained modelin çalıştığını veya
hallucination başarısını göstermez. Önbellek yoksa yalnız iki opsiyonel test atlanır.

Gerçek pretrained encoder ile opsiyonel CPU kontrolü (ağırlıklar yaklaşık
600 MB; standart unit testler bunları indirmez):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('answerdotai/ModernBERT-base', revision='8949b909ec900327062f0ebf497f51aef5e6f0c8', allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors'], local_dir='.cache/modernbert-pretrained')"
python scripts/smoke_transformer.py --model-dir .cache/modernbert-pretrained
```

Script yalnız iki sentetik cevapta bir SGD adımı, JSON dönüşümü ve checkpoint
roundtrip yapar. Dataset okumaz, tam eğitim başlatmaz. Çıktı varsayılan olarak
`artifacts/pretrained-cpu-smoke/` altındadır; mevcut koşumun üzerine yazmaz.
Yeniden denemek için `--output artifacts/pretrained-cpu-smoke-2` kullanılabilir.

## Veri hazırlığı ve 1. kişinin GPU koşumu

1. kişi ayrı P2 ortamında GPU'suna uygun PyTorch'u **önce** kurar. Mevcut ROCm
torch'u CPU wheel ile değiştirmeyin. Requirements `torch>=2.6,<3` koşulu
kurulu uyumlu GPU derlemesini korur; CUDA/ROCm kurulumunu kendisi seçmez.
ROCm'de de PyTorch cihaz adı `cuda`dır. Çalışan ortam kaydı saklanır:

```powershell
python -m pip install -r requirements-transformer.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/download_data.py
python scripts/prepare_data.py
```

Önceden sınırlandırılmış gerçek model pilotu **yalnız 1. kişinin PC'sinde**:

```powershell
python scripts/transformer.py train --config configs/transformer.json --device cuda --limit-train 8 --limit-validation 4 --output outputs/transformer-pilot
```

Bu komut config'teki 3 epoch'u yalnız sınırlı örneklerde çalıştırır. Model
ağırlıklarını ilk kez indirir. GPU belleği/süre ölçümü ve gerekirse epoch,
max_length, batch/precision ayarları 1. kişi tarafından pilotta yapılır.

Tam GPU eğitimi, güncel Git clone içinde merkezî kayıtla:

```powershell
python scripts/run_job.py configs/jobs/transformer_train.json
```

Doğrudan aynı eğitim komutu (merkezî kuyruk/log kaydı sağlamaz):

```powershell
python scripts/transformer.py train --config configs/transformer.json --device cuda --output outputs/transformer-full
```

CPU için `--device cpu` kullanılır; yalnız işlev desteğidir, tam CPU eğitimi
önerisi değildir. `run_job` Git kimliği ister; ayrı clone edilen Git deposundan çalıştırılır.

## Tahmin ve checkpoint

Eğitim sonunda `transformer_summary.json`, `validation_predictions.jsonl` ve
iyileşen epoch'ların `checkpoints/epoch-N/` klasörleri oluşur. Summary seçilen
checkpoint yolunu, config'i, veri/kod/checkpoint hash'lerini, ortamı, sayımları
ve train/validation loss'larını saklar. `run_job` ayrıca Git/koşum kaydını tutar.

```powershell
$runPath = 'outputs/transformer-full'
$runSummary = Get-Content "$runPath/transformer_summary.json" -Raw | ConvertFrom-Json
$checkpointPath = Join-Path $runPath $runSummary.best_checkpoint
python scripts/transformer.py predict --checkpoint $checkpointPath --split validation --device cuda --output outputs/transformer-validation.jsonl
```

Checkpoint: `encoder/` HF config+safetensors, `tokenizer/`, `head.safetensors`,
`checkpoint.json`. Tahminde ağ bağlantısı gerekmez; kaynak model yeniden
indirilmez. Eğitimdeki threshold/aggregation/tokenizer/pencere ayarları korunur.
CPU tahminde BF16 yerine FP32 kullanılır. Mevcut çıktıların üzerine yazılmaz.

Checkpoint biçimi v2'dir. Encoder, tokenizer ve sınıflandırıcı dosyalarının
SHA-256 değerleri `checkpoint.json` içinde saklanır; eksik, eklenmiş veya
değişmiş dosyalar model yüklenmeden reddedilir. Label sırası da doğrulanır.
İlk ZIP teslimindeki checksum içermeyen v1 checkpoint'ler yeni yükleyicide
reddedilir; orijinal v1 kodu ZIP klasöründe korunmuştur. Tam eğitimden üretilmiş
bir v1 checkpoint bu çalışmada bulunmuyordu; hiçbir checkpoint silinmedi.

Bu checkpoint **tahmin için tam model kaydıdır**. Optimizer/RNG/epoch ortası
devam durumu saklanmaz; kesilen eğitime birebir resume desteklendiği iddia edilmez.

Her cevap için bir JSONL satırı:

```json
{"id":"response-id","spans":[{"start":12,"end":20,"label_type":"Evident Conflict"}]}
```

Farklı türler örtüşebilir. Boş tahmin `spans: []` olur. Tür başına komşu
pozitif bölgeler ortak decoder'a göre birleşir; aradaki yalnız boşluklar
span'i bölmez, kenardaki boşluklar atılır. Aynı türdeki anotasyon kimlikleri
ayrı ayrı yeniden üretilmez; token içi anotasyon sınırı token kenarına genişleyebilir.
Gold dosyaları korunur; tahminde gold alanlarına erişilmez.

Resmî test doğrudan CLI ile çalıştırılamaz. Protokol dondurulup commit edildikten
sonra `run_job` üzerinden tam komut ve veri/manifest/checkpoint dosyaları hash'leri
doğrulanmalıdır. Test için limit bayrağı kabul edilmez; hazır final-test job'u
veya protokol onayı bu teslimin parçası değildir.

## Test sonucu ve sınırlar

31 Ekim 2026 güncel main tabanlı clone'daki Windows CPU kontrolü:
**95 test; 92 geçti, 3 atlandı, 0 hata**. Transformer testlerinin **20'si geçti**.
Atlanan üç mevcut test ham/işlenmiş tam RAGTruth dosyalarını gerektiriyor;
bu dosyalar yerelde yok. CRF, veri hizalama,
split, indirme, koşum ve multi-hot regresyon testleri geçti. `pip check` temiz.
CLI yardım komutları çalıştı. CI workflow eklendi; GitHub üzerinde henüz koşmadı.

PR incelemesinde iki yeni regresyon testi önce başarısız oldu: manifest grup
sızıntısının tam kontrolü ve checkpoint yüklemeden checksum doğrulaması eksikti.
İkisi de Transformer tarafında düzeltildi ve testler geçti. Son koşumda başarısız
test yok. Koşum testlerinin logundaki
`status: failed` satırları kasıtlı negatif senaryolardır; unittest başarısızlığı değildir.

Gerçek pretrained ModernBERT-base ile CPU FP32 kontrolü geçti: 22 katman,
149.014.272 encoder parametresi, `[2,19]` girdi, `[2,19,4]` logits, 13 cevap
tokenı, tek optimizer adımı. Embedding ve son attention ağırlıklarının indirilen
checkpoint'ten geldiği birebir karşılaştırıldı. Gradientler sonlu, loss dışında
logit gradientleri sıfır; checkpoint roundtrip logits'leri birebir aynı.
Sınıflandırıcı başlığı yeni ve henüz eğitilmemiştir; bu bir başarı ölçümü değildir.

CUDA/ROCm, 2048-token tam encoder forward/backward, GPU belleği/süre, tam veri
eğitimi, ortak nihai scorer ve resmî test değerlendirmesi **çalıştırılmadı**.
F1/başarı veya GPU'ya sığma iddiası yok.

## Git/PR teslimi

PR öncesi incelemede mevcut ZIP korunarak GitHub deposu ayrı klasöre clone edildi.
`feature/ragtruth-transformer`, `main` / `origin/main` commit'i
`987a18bd6515ba0e3370b582c39c342626423fe5` üzerinden oluşturuldu. Son fetch'te
origin/main aynı commit'teydi. Testler bu clone'da tekrar çalıştırıldı.
README'nin main içeriği korunup yalnız Transformer paragrafı eklendi.
CRF, ortak veri kodları ve koşum scripti değişmedi. Değişiklikler
`feature/ragtruth-transformer` branch'inden `main` hedefli PR teslimi için hazırlandı.

Yerel clone: `C:\Users\Adnan\Desktop\ragtruth-hallucination-detection-main\ragtruth-hallucination-detection`.
Orijinal ZIP kodları yanındaki `ragtruth-hallucination-detection-main` klasöründe
değiştirilmeden durur; ayrıca checksum'lı yedek ZIP alındı.

PR başlığı/açıklaması: [TRANSFORMER_PR.md](TRANSFORMER_PR.md).
**Kullanıcı commit, branch push ve main hedefli PR teslimine onay verdi;
merge işlemi bu teslimin kapsamında değildir.**

## Ekipte netleştirilecekler

- 1 + 4: gerçek GPU pilotu, 2048/1024 seçimi, batch, BF16/FP32, epoch bütçesi;
  CUDA/ROCm paket sürümleri ve süre/bellek kaydı.
- 2 + 4: `best` eğitim / bütün pencerelerle validation asimetrisi, `min` yanlış
  negatif riski, uzun cevaplarda tekrar ağırlığı ve aynı tür span birleşmesi.
- 4 + 5: ortak scorer, boşluk politikası, F1 ile checkpoint/eşik seçimi ve
  küçük validation araması. Şu an seçim masked BCE, eşik sabit 0.5.
- 1 + 2 + 5: quality/implicit_true/due_to_null politikası ve donmuş final-test
  protokolü. Mevcut hedefler değiştirilmedi, anotasyonlar silinmedi.
- 3 + 4: yalnız-cevap kontrolü opsiyoneldir; bu çekirdek teslimde eklenmedi.
- Eğitim sonuçları geldikten sonra 4. kişinin gerçek hata analizi ve slaytları
  hazırlanır; mevcut sentetik testten başarı/hata örnekleri uydurulmaz.
