# 5 Kasım 2026 — değerlendirme geri bildirimi ve teslim düzeltmesi

Bu kayıt, ilk teslimden **sonra** yapılan düzeltmeleri açıkça ayırır. Bir
eksikliğin şimdi giderilmesi, ilk teslimde mevcut olduğu anlamına gelmez.
Veri özgünlüğü puanına konu olan hazır veri seçimi değiştirilmez; yeni veri
toplama, yeni insan anotasyonu veya kapsamlı ham veri temizliği iddiası yoktur.

## Geri bildirim ve karşılığı

| Geri bildirim | Düzeltme/kanıt | Durum ve sınır |
|---|---|---|
| Ham ve işlenmiş veri Git'te yok | `data/raw/` ve `data/processed/` teslimi; [veri dizini rehberi](../data/README.md); `scripts/verify_submission_data.py` | Mevcut byte'lar ve sabit hash'ler korunarak teslim edilir; özgün veri üretimi sayılmaz |
| Kendi kodumuza ait LICENSE yok | Kök [LICENSE](../LICENSE): MIT, © 2026 kaanksn1 and project contributors | Veri lisansından ayrı kod lisansı; upstream hak bildirimleri korunur |
| Metrik seçimi sınıf dengesizliğiyle yeterince gerekçelendirilmemiş | [METRIC_RATIONALE.md](METRIC_RATIONALE.md) ve README'de görünür açıklama | Çoğunluk `O` örneği, dört türün desteği, macro/micro tercihi ve sınırlılıkları; eski protokol değiştirilmez |
| README bazı dosyaları açıklamıyor | [README / Proje yapısı](../README.md#proje-yapısı) | `AGENTS.md`, `PROJECT_HANDOFF.md`, `.github/`, `evaluate.py`, `plot_results.py`, `transformer.py` ve diğer kök öğeler açıklanır |
| Ayrı BERT, HMM, RNN ve CNN benchmark'ları yok | [Tamamlanmış dört yeni baseline ve yedi model karşılaştırması](../results/supplement_2026-11-05/README.md) | Aynı 12.702 train / 2.240 validation; gerçek tahminler ve ölçülmüş skorlar teslim edilir; yeni resmî test sonucu değildir |

## Metrik gerekçesindeki tarihsel kanıt

Teslim öncesi [PROTOCOL_DRAFT.md / Metrikler](PROTOCOL_DRAFT.md#metrikler)
bölümünde, sorunsuz tokenların çoğunluk olabilmesi nedeniyle accuracy'nin ana
ölçüt yapılmayacağı belirtiliyordu. [DATA_REPORT.md](DATA_REPORT.md) eğitimde
anotasyonlu kelime oranını yaklaşık %5,6; dört türün anotasyon sayılarını
4.017 / 160 / 4.654 / 2.006 olarak raporluyordu.

Bununla birlikte ana README ve sunumun, macro F1'a eşit tür ağırlığı verilmesini
dengesizlikle yeterince görünür ilişkilendirdiği iddia edilmez. Yeni belge bu
eksik açıklamayı tamamlar; teslim öncesindeki belgeymiş gibi tarihlendirilmez.

## Tamamlayıcı baseline'ların raporlama sınırı

İlk V1 resmî testinde kaynaklı/yalnız-cevap CRF ve ModernBERT bulunuyordu.
ModernBERT'in BERT ailesinde olması, ayrı klasik BERT baseline'ının
uygulandığı anlamına gelmez. HMM/RNN/CNN/BERT ekleri değerlendirme geri bildirimi
sonrasında yapılan yeni deneyler olarak tanımlanır.

Yeni modeller aynı özgün train/validation bölümlerini ve `quality=good`
politikasını kullanmalıdır. Vocabulary, özellikler, epoch/eşik seçimi ve
hiperparametreler resmî test etiketlerine bakılarak ayarlanmaz. Raporun her
model için girdi türünü, eğitim kapsamını/bütçesini, seed'i, seçim kuralını,
parametreleri, gerçek tahminleri ve ölçülmüş metrikleri belirtmesi gerekir.
Kodun yazılmış olması veya sentetik testin geçmesi, tam veri benchmark'ı
tamamlandığı anlamına gelmez.

Yeni modellerin tümü aynı kaynak bağlamını kullanmıyorsa mimari farkı tek
başına bağlam etkisi veya salt mimari üstünlük diye yorumlanmaz. Sadece
validation'da ölçülen yeni sonuçlar resmî test skoru diye etiketlenmez.
Tarihsel test zaten görülmüş olduğundan ek çalışma özgün deneyden sonra
yapılmıştır; ilk çalışmanın önceden dondurulmuş karşılaştırması gibi sunulmaz.

## Korunan tarihsel sonuçlar

`docs/FINAL_PROTOCOL.json`, `configs/final/` ve `results/test_comparison.*`
eski çalışmanın referansıdır. Eski test tablosu, checkpoint/eşik seçimi veya
veri bölümleri bu teslim düzeltmesiyle yeniden tanımlanmaz. Mevcut V1
Transformer'ın resmî any-character F1'ı **0,5206157437**'dir; bu accuracy veya
response-level F1 değildir. [Teknik rapor](../results/FINAL_REPORT.md) ve
[2 Kasım bağımsız doğrulaması](../results/review_2026-11-02/REVIEW.md) korunur.

## Kontroller ve son durum

Veri kimliği kontrolü: `python scripts/verify_submission_data.py`.
Kod kontrolleri: `python -m unittest discover -s tests -v`.
Bağımlılığı bulunmayan ortamda atlanan testler tam kapsam sayılmaz.

Dokuz veri dosyası / **278.308.446 byte**, hem yerel byte/hash kontrolünden hem
gerçek Git blob denetiminden geçti. Dört yeni modelin gerçek kayıtları ve
yedi modelin ortak değerlendirmesi [ek rapordadır](../results/supplement_2026-11-05/README.md).
HMM/RNN/CNN metrikleri birebir yeniden üretildi; BERT'in tam kohortu, checkpoint
hash'leri ve kalibrasyon metrikleri doğrulandı. Yerel iki ortamda **272 test**
çalıştı: HF/CPU **269 geçti, 3 atlandı**; native/CRF **234 geçti, 38 atlandı**.
İlk kod teslimi `4302236` için dört GitHub kontrolü geçti; güncel teslimin
kontrolleri GitHub pull request kaydındadır.
