# RAGTruth Halüsinasyon Tespiti

**NLP Proje 2 · Grup Anonymous**

Bu proje, kaynak metne dayanarak üretilen LLM cevaplarında kaynakla çelişen veya
kaynakta desteklenmeyen bölümleri karakter düzeyinde bulur ve dört türe ayırır.
Amaç genel bir doğruluk kontrolü değil, cevabın kaynağa bağlılığını ölçmektir.

**5 Kasım 2026 teslim düzeltmesi:** Ham/işlenmiş veri teslimi, proje kodunun
lisansı, dosya açıklamaları ve metrik gerekçesi [düzeltme kaydında](docs/SUBMISSION_FIXES_2026-11-05.md)
izlenir. Sonradan eklenen baseline'lar ayrı bir ek çalışmadır; aşağıdaki tarihsel
resmî test sonuçları değiştirilmez.

## Veri

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) ([ACL 2024](https://aclanthology.org/2024.acl-long.585/)),
2.965 kaynak ve 17.790 LLM cevabı içerir. Dört tür: Evident Conflict, Subtle Conflict,
Evident Baseless Info ve Subtle Baseless Info.

Train ve validation kaynak grubuna göre ayrıldı; aynı kaynak iki tarafta birden yer almaz.
Ham kaynak dosyaları `data/raw/`, hazırlanmış kayıtlar `data/processed/` altında
teslim edilir; verinin kökeni ve yeniden üretim adımları [veri dizini rehberinde](data/README.md)
açıklanır. `python scripts/verify_submission_data.py` orijinal ve hazırlanmış
dosyaların sabit SHA-256 değerlerini denetler.

RAGTruth hazır ve insan tarafından etiketlenmiş bir araştırma veri setidir.
Bizim katkımız veri birleştirme, kalite filtresi, offset doğrulama, sızıntısız
bölümleme ve model girdisine hizalamadır; yeni bir corpus veya yeni insan
anotasyonları üretildiği iddia edilmez. Ayrıntılar: [veri raporu](docs/DATA_REPORT.md).

## Yöntem

| Model | Yaklaşım |
|---|---|
| CRF (yalnız cevap) | Her tür için ayrı linear-chain CRF, kaynak özelliği yok |
| CRF (kaynaklı) | Aynı model, kaynak özellikleriyle |
| Transformer | ModernBERT-base, dört bağımsız çıktı, en fazla 2.048 token |

Dört bağımsız çıktı, farklı türdeki örtüşen etiketleri korur. Model ve eşik seçimi
validation ile yapıldı; kararlar resmî testten önce [donduruldu](docs/FINAL_PROTOCOL.json).

İlk teslimde ayrı klasik BERT, HMM, RNN ve CNN benchmark'ları bulunmuyordu.
Bu dört modelin tam veri koşumları tamamlandı; gerçek tahminler, skorlar ve
sınırlılıklar [yedi modelin validation benchmark'ında](results/supplement_2026-11-05/README.md)
ve [teslim düzeltmesi kaydında](docs/SUBMISSION_FIXES_2026-11-05.md) açıklanır.
ModernBERT'in BERT ailesinden olması, ayrı bir klasik BERT karşılaştırmasının
yerini aldığı anlamına gelmez.

## Metrik seçimi ve sınıf dengesizliği

Eğitim verisindeki kelime tokenlarının yalnız yaklaşık %5,6'sı anotasyonludur;
sürekli `O` tahmini yüksek accuracy verirken hatalı bölgeleri hiç bulmayabilir.
Dört türün eğitim anotasyon sayıları da **4.017 / 160 / 4.654 / 2.006**'dır.
Bu nedenle donmuş V1 Transformer seçiminde **dört türün character macro F1 ortalaması**
birincil, **any-character micro F1** eşitlik bozucu ölçüttür. Macro her türe
eşit ağırlık verir; nadir bir türün başarısızlığı çoğunluk içinde kaybolmaz.

Tarihsel CRF ayar seçimi any-character F1 kullanmıştı. Bu tercih ve farklı
arama bütçeleri açıkça korunur; bütün modellerin aynı seçim kuralıyla eşit
ölçüde optimize edildiği iddia edilmez.

Any-character F1 genel bölge tespitini; exact typed-span F1 ise hem sınırın hem
türün tam eşleşmesini ölçer. Precision/recall ve tür destekleri de raporlanır.
Macro, az sayıdaki örneğe duyarlıdır; tek başına yüksek olması genel tespitin
yüksek olduğu anlamına gelmez. Gerekçe ve formüller:
[metrik gerekçesi](docs/METRIC_RATIONALE.md) · [değerlendirme sözleşmesi](docs/EVALUATION.md).

## Sonuçlar

Resmî test, `quality=good` olan 2.675 cevap üzerinde değerlendirildi:

| Model | Any-character F1 | Typed macro F1 | Exact typed-span F1 | Response-level F1* |
|---|---:|---:|---:|---:|
| CRF (yalnız cevap) | 0,2074 | 0,1195 | 0,0615 | 0,5878 |
| CRF (kaynaklı) | 0,3544 | 0,1976 | 0,0667 | 0,6624 |
| Transformer | **0,5206** | **0,2984** | **0,0678** | **0,7285** |

\* Cevap düzeyi: cevapta en az bir hatalı span var/yok kararı. Sonradan eklenen
açıklayıcı metriktir; model seçiminde kullanılmadı ve karakter F1'la aynı görev değildir.

Sabit tahminlerin skorları, dosya kimliği ve tam kohort kontrolüyle proje scorer'ı
dışında [bağımsız olarak yeniden üretildi](results/independent_rescore.md).

Modeller sorunlu bölgeleri kısmen buluyor, ancak span sınırlarını tam tutturmakta
zorlanıyor. Subtle Conflict F1 üç modelde de 0'dır; bu tür testte yalnızca 15 cevapta
geçer. Sonuçlar tek seed'e dayanır.

Ayrıntılar: [teknik rapor](results/FINAL_REPORT.md) · [Türkçe bulgular](results/BULGULAR_TR.md) ·
[hata analizi](results/error_analysis.md) · [etiket kalitesi](results/label_quality.md)

## Son geliştirmeler

- **Eksik baseline'lar tamamlandı:** Aynı 2.240 validation cevabında any-character
  F1: HMM **%21,02**, RNN **%18,08**, CNN **%18,99**, klasik BERT **%41,16**.
  Tarihsel CRF ve ModernBERT ile [ortak karşılaştırma](results/supplement_2026-11-05/README.md),
  tahmin dosyaları ve koşum kimlikleri Git'tedir. Bunlar geri bildirim sonrası
  validation ekleridir; eski resmî test tablosunun yerine geçmez.
- **Yeni seçim deneyleri:** Ayrı oturumda any-F1 öncelikli seçim ve ortak eşik
  araması denendi. Raporlanan validation F1: Transformer %55,07, CRF %39,22.
  Bunlar bağımsız test sonucu değildir; yeni oturumun tahmin/ağırlıkları bu depoda yoktur.
  Teslim edilen Transformer calibration JSON'u global %55,00 ayarını içerir;
  %55,07 joint sonucu ayrı deney raporundadır. Eski donmuş test sonuçları değişmedi.
  Tarihsel macro-F1 seçim modu korunur; yeni any-F1 modu ve güncel komutlar
  [kalibrasyon rehberinde](docs/CALIBRATION.md) açıklanır.
- **Doğrulama düzeltmeleri:** Eksik/tekrar tahminler ve değişmiş girdiler reddedilir;
  test raporu açık izin gerektirir. Etiket denetimi yalnız train/validation okur.
- **Bağımsız doğrulama:** Test skorları yeniden üretildi ve etiket kalitesi denetlendi.
  Düzeltmeler ve kontroller: [2 Kasım inceleme raporu](results/review_2026-11-02/REVIEW.md).
- **V2 denemeleri:** Focal loss, BCE ve yapılandırılmış modeller ayrı bir iç holdout ile
  [`feature/ragtruth-v2`](https://github.com/kaanksn1/ragtruth-hallucination-detection/tree/feature/ragtruth-v2)
  branch'inde tutulur. Bu sonuçlar resmî testle doğrudan karşılaştırılamaz.

## Kurulum

Python 3.12 gerekir. Veri indirme/hazırlama standart kütüphane ile çalışır:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

Model bağımlılıkları `requirements-crf.txt` ve `requirements-transformer.txt` içindedir.
Grafikler için `requirements-evaluation.txt`, RNN/CNN için
[`requirements-baselines.txt`](requirements-baselines.txt) kullanılır; HMM standart
kütüphane ile çalışır. Yeni baseline komutları: [HMM](docs/BASELINE_HMM.md),
[RNN/CNN](docs/BASELINE_NEURAL.md), [klasik BERT](docs/BASELINE_BERT.md).
Tam test kapsamı için bu bağımlılıkları uygun ortamda kurup
`python -m unittest discover -s tests -v` çalıştırın. Bağımlılık bulunmayan ortamda
bazı testler atlanır; yeşil sonuç bütün model kontrollerinin çalıştığı anlamına gelmez.
Eğitim ve değerlendirme adımları: [çalıştırma rehberi](docs/RUNNING.md) ·
[teknik teslim rehberi](docs/TECHNICAL_HANDOFF.md).

## Proje yapısı

| Yol | İçerik |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Kodlama asistanları için sahiplik, veri/test politikası ve açık görev atamaları; model girdisi değildir |
| [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) | Ayrı bir çalışma oturumunun tarihsel devir notu; başındaki kanıt sınırı ve güncelleme okunmalıdır |
| [`.github/`](.github/) | Pull request şablonu ve Python/Transformer GitHub Actions test iş akışları |
| [`evaluate.py`](evaluate.py) | Sabit tahminleri ortak karakter/span metrikleriyle karşılaştıran CLI; model eğitmez |
| [`plot_results.py`](plot_results.py) | Ortak değerlendirme JSON'undan karşılaştırma grafikleri üretir |
| [`transformer.py`](transformer.py) | ModernBERT ve ek klasik BERT için ortak token sınıflandırıcısı, konfigürasyon ve eğitim/tahmin yardımcıları; CLI `scripts/transformer.py` içindedir |
| [`neural_baselines.py`](neural_baselines.py) | Geri bildirim sonrası eklenen kelime düzeyinde RNN/CNN, train-only sözlük ve modeli yeniden yükleme yardımcıları |
| [`data/`](data/README.md) | Ham veri, hazırlanmış kayıtlar, kaynak gruplu manifest/hash'ler ve küçük örnekler |
| `ragtruth_data/` | Veri şeması, bölümleme ve hizalama |
| `ragtruth_eval/` | Ortak değerlendirme |
| `scripts/` | Veri, eğitim, kalibrasyon ve analiz komutları |
| `configs/` | Model, deney ve dondurulmuş koşum ayarları |
| `docs/` | Veri/model sözleşmeleri, metrik gerekçesi, koşum ve teslim belgeleri |
| `demo/` | Yerel sonuç görüntüleyici |
| `results/` | Raporlar, metrikler ve grafikler |
| `tests/` | Birim ve entegrasyon testleri |
| `contributions/` | Üye katkı dosyaları |
| `requirements-*.txt` | CRF, Transformer ve değerlendirme ortamlarının ayrı bağımlılık listeleri |
| [`requirements-baselines.txt`](requirements-baselines.txt) | Ek RNN/CNN baseline'larının Torch bağımlılığı; HMM standart kütüphane kullanır |
| [`.gitignore`](.gitignore) | Ağırlıklar, kişisel ortamlar, cache ve geçici çıktılar için hariç tutma kuralları; teslim veri dosyaları açıkça izinlidir |
| [`.gitattributes`](.gitattributes) | Metin satır sonları ve donmuş veri/konfigürasyon byte'larının korunması |
| [`LICENSE`](LICENSE) | Bu projenin kendi kaynak kodunun MIT lisansı; RAGTruth veri lisansından ayrıdır |

`outputs/`, `artifacts/`, `.cache/`, `.venv/` ve `__pycache__/` çalışma anında
oluşan yerel dizinlerdir; bunların tamamı Git teslimine dahil değildir. Küçük,
doğrulanmış raporlar `results/` altında bulunur. Ağırlık dosyalarının Git'te
bulunmaması, `data/raw/` ve `data/processed/` altındaki teslim dosyalarının
bulunmadığı anlamına gelmez.

## Ekip

| Üye | Katkı |
|---|---|
| Serdar Kaan Kesen | Merkezî koşum, entegrasyon, değerlendirme ve test |
| Samed Ortakaya | Veri sözleşmesi, bölümleme ve hizalama |
| Kemal Arif Karabacıoğlu | CRF baseline |
| Adnan Demir | Transformer ve seçim düzeltmeleri |
| Ozan Serhan Denizhan | Bağımsız doğrulama, etiket denetimi ve sunum |

Ayrıntılar: [katkı dosyaları](contributions/README.md).

## Lisans

**Proje kaynak kodu:** [MIT — LICENSE](LICENSE), © 2026 kaanksn1 and project contributors.

**RAGTruth verisi:** Particle Media'nın ayrı MIT bildirimi
[`data/raw/LICENSE`](data/raw/LICENSE) ve [kopyası](docs/DATA_LICENSE.txt).
Ham ve türetilmiş veri bu köken/atfı korur. Önceden eğitilmiş model ağırlıkları
burada yeniden lisanslanmaz veya Git'e eklenmez; kendi upstream koşullarına tabidir.
