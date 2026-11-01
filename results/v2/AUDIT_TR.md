# Proje 2 ayrıntılı teknik denetimi — 1 Kasım 2026

**Sonuç:** Eğitimler ve sabit değerlendirmeler tamamlanmış. İncelenen veri,
etiket/offset, metrik ve checkpoint kontrollerinde sonuçları geçersiz kılan
bir bütünlük hatası bulunmadı. Buna rağmen yöntem ve teslim tarafında aşağıdaki
eksikler var. Teslim/paket sorunları giderildi; model başarısının sınırlamaları
aynen korunup açıkça raporlandı. Yeni eğitim/eşik araması yapılmadı.

## Öncelikli bulgular

| Öncelik | Bulgu ve etkisi | Durum |
|---|---|---|
| Yüksek | Kullanıcıya anlatılan ana hedef any-character F1 iken seçim dört-tür macro F1'ı önceliklendirdi. Aynı focal epoch-2'de calibration any F1'ı %51,22 → %44,96 düşürdü. | Yöntem sınırlaması; mevcut seçimi/testi sonradan değiştirmedik. |
| Yüksek | Structured decoder'da nadir Subtle Conflict türü için çok küçük macro kazancı, çok büyük yanlış alarmı kabul ediyor: dev'de 830 gold karaktere karşı 1.843.432 tahmini karakter. | Gerçek hata davranışı; modelin kullanılabilirliği sınırlı. Negatif deney olarak raporlandı. |
| Yüksek | İlk V2 ZIP'inde kök `structured_transformer.py` yoktu. V2 split dosyaları gerçek çalışma yollarında değildi; `results/v2` düzleştirilmişti ve bazı rapor bağlantıları kırılıyordu. | Yeni `package_v2_handoff.py` ve regression testleriyle giderildi. Eski ZIP değiştirilmedi. |
| Orta | Fresh clone için V2 kaynak-veri, cache, makineye özel supervisor ve CLI varsayılanları yeterince açıklanmamıştı. Bazı eski durum metinleri hâlâ “eğitim başlamadı” diyordu. | [V2 rehberi](../../docs/V2_REPRODUCIBILITY.md), güncel durum ve açık yollar eklendi. |
| Orta | İlk V2 grafik başlıkları yalnız `test` diyordu; ayrı gösterildiğinde resmî testle karışabilir. | Sayılara dokunmadan `V2 internal holdout` başlıklı ek grafikler üretildi. |
| Orta | Epoch checkpoint'lerinde optimizer/RNG/yarım epoch konumu yok; supervisor genel resume desteklemiyor. | Gelecek eğitimler için kısıt. Tamamlanmış dokuz checkpoint ve sonuçlar mevcut. |
| Orta | Tek seed; güven aralığı/çoklu seed yok. Nadir sınıfta dev desteği az; eşik seçimleri kararsız olabilir. | Küçük farkların istatistiksel kazanım olduğu iddia edilmiyor. |
| Orta | Yeni holdout'u hazırlayan manuel kapı, commit edilmiş `status=frozen` alanını kontrol ediyor; tek başına bütün protokol kimliklerini doğrulamıyor. | Kullanılan merkezi pipeline ayrıca kaynak/ağırlık/veri hash'lerini ve frozen job'ları doğruladı. Manuel CLI daha zayıf; mevcut sonucu bozduğuna dair kanıt yok. |

İlk iki bulgunun aday bazında sayısal açıklaması
[DIAGNOSIS_TR.md](DIAGNOSIS_TR.md)'dedir. “CRF ekledik, kesin iyileştirir” veya
“biraz daha epoch ile %80 olur” sonucunu destekleyen kanıt yoktur.

## Bağımsız veri denetimi

| V2 bölümü | Toplam cevap | `good` cevap | Kaynak | Kaynak grubu |
|---|---:|---:|---:|---:|
| Eğitim | 11.550 | 11.431 | 1.925 | 1.923 |
| Doğrulama / dev | 2.262 | 2.240 | 377 | 377 |
| Yeni internal holdout | 1.278 | 1.271 | 213 | 213 |

Üç bölümün her ikili karşılaştırmasında cevap kimliği, kaynak kimliği, kaynak
grubu, tam kaynak metni ve casefold + boşluk normalizasyonlu kaynak metni
örtüşmesi **0**. Bu kontrol anlamsal olarak benzer bütün farklı kaynakları
bulduğumuz anlamına gelmez. Ortak cevap kalıpları ayrı raporlandı; farklı
kaynaklardaki bu tekrarların gold halüsinasyon span'i yok. “Hiç tekrar yok”
iddiası yapılmıyor.

V2'deki bütün kayıtlar yalnız orijinal train/dev kayıtlarına karşı eşlendi:
cevap/kaynak metni, dört tür, örtüşen span'ler ve metadata değişmemiş;
yalnız split/study alanları farklı. Gold sınırları ve varsa span metinleri
karakter konumlarıyla uyumlu. Orijinal resmî test anotasyon dosyası açılmadı.

Kalite filtresi gerçek değerlendirme kohortunda aynı: dev'de 19 yanlış ret +
3 kesilmiş cevap, yeni holdout'ta 5 yanlış ret + 2 kesilmiş cevap hariç.
`implicit_true` ve `due_to_null` gibi etiket metadata'sı silinmiyor; bunlar
modele girdi olarak verilmedi. Sayılar farklı kalite alt kümeleriyle karıştırılmamalı.

Bu denetim anotasyonların **biçimsel doğruluğunu** kontrol eder. Her bir insan
etiketinin anlamsal olarak tartışmasız doğru olduğunu kanıtlamaz. Hazır insan
anotasyonları kullanıldı; ekip tarafından yeni gold anotasyon üretildiği veya
bağımsız iki anotatörle tüm veri yeniden incelendiği iddia edilmemeli.

## Metrik ve ağırlık doğrulaması

Kaydedilmiş üç model × iki V2 kümesi için karakter kümeleriyle bağımsız scorer
çalıştırıldı. Any, dört tür, typed-micro, fixed-four macro ve ham sınırlı exact
span precision/recall/F1 ile TP/FP/FN sayıları kayıtlarla aynı (`1e-12` tolerans).
Aynı karşılaştırma görev ve generator kırılımlarında da yapıldı: her model/küme
için 3 görev + 6 generator. Bu işlem yeni model tahmini veya yeni seçim değildir.

Karakter metrikleri Unicode code point kullanır ve span içi boşlukları sayar.
Exact typed-span F1, sınırları **ve türü** tamamen eşleşen span'leri sayar;
accuracy veya cevap düzeyinde exact match değildir. Bu nedenle any F1 ile aynı
sayının çıkması beklenmez. V1 makalesi/başka benchmark metrikleriyle ancak aynı
protokol sağlanırsa doğrudan karşılaştırılabilir.

V2 dondurulmuş **97 girdi + 24 seçili artifact** SHA'sı eşleşti. Ayrıca üç
modelin her birinin üç epoch'undan **dokuz checkpoint'in** iç dosya hash'leri
doğrulandı. V1 ana deponun 97 dondurulmuş dosyası da değişmemişti. Denetimde
eğitim kodu, protokol, ağırlık, eşik ve sabit tahmin dosyaları değiştirilmedi.

Dev'deki gold etiketler kelime/BIO biçimine dönüştürülüp tekrar span'e
çevrildiğinde any-character F1 **%99,999**, exact typed-span F1 **%99,70**.
Bu **gold kullanan temsil teşhisidir, model başarısı değildir**; bir optimum
başarı garantisi veya bütün tokenizer'lar için üst sınır değildir. Structured
modeldeki düşük sonucun yalnız kelime/BIO biçiminin zorunlu bilgi kaybıyla
açıklanamayacağını gösterir. Modelin ürettiği sınır ve türler gerçekten zayıf.

## Başarı iddiasının doğru kapsamı

| Yeni internal holdout | Any character F1 (%) | Dört-tür macro F1 (%) | Exact typed-span F1 (%) |
|---|---:|---:|---:|
| BCE — dev ile seçilen birincil | 50,14 | 27,23 | 7,68 |
| Focal — karşılaştırma | 42,50 | 29,30 | 2,48 |
| Structured — negatif deney | 9,35 | 12,01 | 6,92 |

Focal'ın bu holdout'ta macro bakımından yüksek çıkması, sonradan onu birincil
seçmek için kullanılmadı. V2 iyileştirme turu any F1 artışı sağlamadı. Eski
resmî testteki %52,06 ile yukarıdaki %50,14 farklı kümelerden gelir; aynı testte
2 puan kayıp/kazanım gibi anlatılmamalı. V1 modelleri yeni holdout'un kayıtlarını
V1 eğitiminde gördüğünden bu yeni kohortta geçerli bir kontrol olarak çalıştırılmadı.

## Testler, demo ve teslim

Unit testler hem Torch/Hugging Face ortamında hem native CRF ortamında çalıştı.
İlgili bağımlılık bulunmayan ortamda bazı testler atlandı; bu yüzden iki ortamın
sonuçları birlikte değerlendirildi. Güncel kesin test sayıları ve paket kontrolleri
[`delivery_audit.json`](delivery_audit.json)'dedir. Denetimden önceki GitHub
Transformer/Python kontrolleri de başarılıydı; yeni commit ayrıca CI'dan geçer.

Demo 2.240 dev cevabı ve üç model içerir. Kaynak/cevap/gold/tahmin paneli ve
Unicode segment bütünlüğü kaydedilmiş dosyalarla kontrol edildi; haricî script
bağımlılığı yok ve MIT veri lisansı gömülü. Bu denetimde tarayıcıdaki bütün
kontrol kombinasyonları için yeni bir etkileşim testi yapılmadı. Demo bir ürün
veya canlı doğrulama servisi olarak sunulmamalı.

Denetlenmiş ZIP, V2 modülünü ve split sözleşmelerini doğru yolda içerir; rapor
klasörleri korunur, CRC ve SHA manifesti doğrulanır. Kod/protokol/ağırlıkların
orijinal dondurulmuş haliyle uyuşması kontrol edilir. Ağırlıkların ve işlenmiş
verinin pakette/GitHub'da bulunmaması açıkça belgelenmiştir.

Slayt ve sunum bu çalışmanın dışında. Teknik anlatımda amaç, hazır anotasyonun
kaynağı, gerçek temizlik/hizalama emeği, source-aware görev oluşu, ayrımın
korunması, üç ayrı metrik ve başarısız deneyler açıkça belirtilmeli. Yeni bir
iyileştirme turu yapılacaksa hedef önce belirlenmeli ve yeni kararlar mevcut
holdout'lara göre ayarlanmamalı.

Kanıt dosyaları: [veri/hash/bağımsız metrik denetimi](integrity_audit.json),
[teslim denetimi](delivery_audit.json), [yeniden üretim rehberi](../../docs/V2_REPRODUCIBILITY.md).
