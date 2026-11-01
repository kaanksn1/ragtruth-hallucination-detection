# PR #9–#11 ve ilişkili kalibrasyon düzeltmeleri — 2 Kasım 2026

İnceleme başlangıcında üç PR da main'e alınmıştı (`f85c412`). Düzeltmeler
ayrı dalda yapıldı. Kişisel katkı dosyaları, slaytlar, eski frozen config'ler,
resmî test sonuçları ve ayrı V2 worktree değiştirilmedi.

## Düzeltilen sorunlar

| Alan | Önceki davranış | Güncel davranış |
|---|---|---|
| Bağımsız scorer kohortu | Prediction ID birleşimi gold kümesini tanımlıyordu; eksik dosya sessizce daha küçük kümede puanlanabilirdi. | Tam raw/manifest/good kohortu sabit referanstan kurulur; her prediction tam aynı ID kümesini içermek zorundadır. |
| Bağımsız scorer kimliği | Değişmiş raw veri, prediction veya kalite kohortu için yeterli bağ yoktu. | Raw revision/SHA, manifest, summary, frozen karar/evaluation referansları ve prediction SHA'ları puanlamadan önce doğrulanır. |
| Varsayılan test erişimi | Script her çağrıda test etiketlerini de okuyordu. | Varsayılan validation; test için ayrı `--split test --allow-test-report-only` gerekir. İzin kapısı veri erişiminden önce çalışır. |
| Sınır tanıları | Noktalama kırpması geçersiz boş span'leri exact eşleşmeye sokabiliyordu. | Boş span'ler çıkarılır; tanılar yalnız dev'de yapılır. Many-to-many örtüşme, bire bir span-F1 diye sunulmaz. |
| Etiket kalitesi denetimi | Girdi kimliği/alan tipleri kontrolü eksik; eksik sınıflar istatistikleri çökertiyordu. | Sabit train/dev SHA'ları, üyelik, sayı, tür, bounds/metin ve boolean flag'ler doğrulanır; boş sınıflar null/0 üretir. |
| Serbest not ayrıştırması | Prose içindeki tür kelimeleri açık başlık gibi yorumlanabiliyordu. | Yalnız açık başlıklar kabul edilir; 24 şüpheli satır unparsed sayılır. Başlık farkı gold hatası veya IAA olarak yorumlanmaz. |
| Precision floor | Hiçbir eşik koşulu sağlamazsa yine en yüksek grid değeri kabul ediliyordu. | Güvenli aday yoksa seçim hata verir; uygunsuz maksimum eşik seçilmez. |
| Kalibrasyon uyumluluğu | Any-first kod eski macro config'lerini reddediyor, merkezi pipeline eski policy'yi taşımaya devam ediyordu. | Açık `legacy_macro` ve `successor_any` modları; eski pipeline/frozen karar macro-first kalır, successor için ayrı job vardır. |
| Transformer decoder kaydı | Joint raporu ile normal calibration artifact'i/provenans ilişkisi eksikti. | Yeni koşumda gerçekten seçilen joint decoder normal artifact olarak dışa aktarılır; politika, producer hash/Git durumu ve süreler kaydedilir. Eski session JSON'ları yeniden yazılmadı. |
| CRF decoder kaydı | Joint kazanan yalnız summary/ayrı tanı dosyasındaydı; normal predict global decoder'ı yüklüyordu. | Yeni kazanan ayrıca `models/source_selected` ve `validation_predictions_selected.jsonl` olarak saklanır; predict per-type eşikleri yeniden yükler. Eski global çıktı ayrı korunur. |
| README/teknik devir | %55,07 raporu uygulanmış/bağımsız doğrulanmış sonuçla karışabiliyor, nadir sınıftan tavan çıkarılıyordu. | Raporlanan dev adayı, teslim edilmiş global decoder ve frozen resmî test ayrıldı. Başarı tavanı iddiası kaldırıldı. |

CRF'de precision tabanı per-type/joint aramalarının kısıtıdır; tarihsel global
diagnostic adayları için veya yeni verilerde evrensel FP garantisi değildir.
Successor Transformer seçimi ise default/global adaylarının da precision
uygunluğunu kontrol eder. Tarihsel macro modu eski bilimsel kararı korumak için
bu yeni kısıtları geriye dönük uygulamaz.

## Gerçek veride doğrulama

- [Validation bağımsız skorları](independent_validation.json): **2.240** good cevap,
  beş kayıtlı V1 modelinin any/macro/exact/per-type metrikleri referansla
  `1e-12` toleransta eşleşti. Resmî test etiketleri decode edilmedi.
- [Resmî test arşiv doğrulaması](independent_test.json): açık rapor izniyle
  **2.675** good cevap ve yalnız önceden dondurulmuş üç prediction dosyası.
  Üç modelin ana skorları aynı çıktı: kaynaklı CRF %35,44, yalnız cevap CRF
  %20,74, Transformer %52,06 any-F1. Yeni model tahmini veya ayar seçimi yok.
- [Etiket denetimi](label_quality.json): **14.942** good train/dev cevabı,
  **12.693** span. 11.812 açık başlık, 881 unparsed, 77 başlık/etiket farkı.
  Aynı source ID ile 43; aynı kaynak metni veya manifest grubuyla 44 tekrar
  çifti var. Hepsi negatif; anotatör uyumu veya pozitif span güvenilirliği
  hakkında ölçüm sağlamaz. Raw/test anotasyon dosyaları bu denetimde açılmadı.

İlk PR JSON'ları tarihsel kayıt olarak tutuldu; yeni sürüm ve girdi SHA'ları
yukarıdaki tarihli raporlarda. Yalnız dev kırpma tanılarında boş span düzeltmesi
nedeniyle küçük farklar var; ana metrik tanımları veya frozen test skorları değişmedi.

## Testler ve sınırlar

`python -m unittest discover -s tests -v` iki ortamda tamamlandı:

| Ortam | Toplam | Başarılı | Atlanan | Hata |
|---|---:|---:|---:|---:|
| Torch/Hugging Face | 248 | 245 | 3 | 0 |
| Native CRF | 248 | 217 | 31 | 0 |

Yeni testler eksik/tekrar ID, kötü span, raw/ref/prediction hash bozulması,
izin kapısı, boş kırpma, eksik sınıflar, yanlış metadata, unsafe eşik,
legacy/successor uyuşmazlığı ve seçili decoder'ın export/predict round-trip'ini kapsar.
Tam model eğitimi veya gerçek-veri eşik araması çalıştırılmadı. Küçük sentetik
train/predict fixture'ları ve mock calibration testleri mevcut test kapsamındadır.

31 korunan protokol/arşiv/katkı dosyası kontrol edildi. Dört tarihsel config/job
ham byte SHA'sı frozen kararla aynı; V2 dalı temiz. Katkı dosyalarındaki 18
commit bağlantısı yerel Git tarihinde mevcut. Bu teknik kaynak kontrolü,
üyeler adına kişisel deneyim/katkı beyanını onaylamak değildir; raporlar değiştirilmedi.

Arkadaşın yeni oturumunun ağırlık, cache ve ham validation prediction'ları bu
checkout'ta bulunmadığından **%55,07 hâlâ raporlanan dev sonucudur**. Eski
global calibration JSON'unu bu değer doğruymuş gibi değiştirmedik. Yeni export
yolunun düzeltilmesi bu eksik dosyaları üretmez ve F1 artışı iddiası değildir.

Güncel komutlar: [bağımsız skor denetimi](../independent_rescore.md),
[etiket denetimi](../label_quality.md), [kalibrasyon modları](../../docs/CALIBRATION.md),
[CRF seçili decoder](../../docs/BASELINE_CRF.md).
