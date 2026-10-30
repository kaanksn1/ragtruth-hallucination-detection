# Anonymous — Proje 2 görev dağılımı

29 Ekim 2026 tarihli kullanıcı kararı: **bütün tam model eğitimleri repo
sahibinin RX 7800 XT / 32 GB RAM bilgisayarında çalışır; 5. kişi sunumu yapar.**
İsimler önceki projenin görev sırasıyla eşleştirildi: 1 Serdar (`kaanksn1`),
2 Adnan (`adnan-demir`), 3 Samed (`SmtOrtakaya`), 4 Kemal (`kemalarif`),
5 Ozan (`serhandenizhan`).
Hocanın zorunlu yöntemleri hakkında ek bir yönerge gelirse kapsam güncellenir.

Paylaşılan ders materyalleri incelendi; mevcut sahiplik dağılımı korunur.
[Dersle eşleştirme notu](LECTURE_ALIGNMENT.md) her kişinin kendi tesliminde
açıklaması gereken hizalama, CRF/Viterbi, loss maskesi ve span metriklerini
belirtir. P1 slaytlarındaki yöntem/süre koşulları P2'ye otomatik taşınmaz.

Bu bir görev ve teslim planıdır; aşağıdaki model/eğitim/değerlendirme
uygulamalarının tamamlandığı anlamına gelmez. Önceden hazırlanmış veri indirme
ve ilk bütünlük kontrolü ortak başlangıç altyapısıdır; herhangi bir üyenin
kişisel veri deneyini tamamladığı gibi sunulmaz.

## 1. kişi — repo sahibi: merkezî koşum, entegrasyon ve demo

**Kod/teknik sorumluluğu:** GPU kontrolü, ortak çalıştırma komutları, eğitim
kuyruğu ve kayıtlar, checkpoint/disk yönetimi, PR entegrasyonu ve sabit
tahminleri gösteren küçük yerel demo.

Teslimleri:

- Çalışma ortamının paket/sürüm kaydı ve tekrar çalıştırılabilir komutları.
- Gerçek modelle kısa GPU pilotu: bellek, hız ve tahmini tam eğitim süresi.
- CRF dahil bütün tam eğitimlerin, validation tahminlerinin ve protokol
  dondurulduktan sonraki son test tahminlerinin merkezî çalıştırılması.
- Her koşum için konfigürasyon, seed, veri kimliği, kod sürümü, süre, log ve
  checkpoint bilgisi. Gereksiz paralel GPU koşumları yerine sıralı kuyruk.
- Kaynak ve cevapla birlikte sorunlu bölümleri/türleri renklendiren küçük
  demo; doğruluk garantisi vermeyen, etiketli gösterim örnekleri.
- Entegrasyon/tekrar üretilebilirlik açıklaması, demo ekranı ve konuşmacı notları.

Eğitimi burada çalıştırmak model kodunun yazarlığını değiştirmez. 3–4. kişiler
kendi modellerinin kodunu ve hatalarını düzeltir; 1. kişi tüm model geliştirme
işini üzerine almaz.

## 2. kişi — veri, ortak örnekler ve hizalama

**Kod/teknik sorumluluğu:** Kaynak–cevap eşlemesi, veri kalite incelemesi,
kaynak gruplarını koruyan train/validation manifesti, anotasyon/kelime/token
offsetleri ve ortak veri çıktıları.

Teslimleri:

- Eğitim verisinde kalite işaretleri, tekrarlar, dört türün dağılımı ve model
  tokenizer'ıyla gerçek uzunluk analizi. Filtreleme varsa gerekçesi ve sayıları.
- Aynı source_id'nin bölümler arasında paylaşılmadığı validation manifesti.
  Farklı ID'lerde aynı kaynak içeriği bulunursa gruplama kararının kaydı.
- Orijinal cevap/karakter aralıklarını değiştirmeyen, dört türü ve örtüşen
  anotasyonları koruyan ortak veri biçimi ve anlamlı hizalama kontrolleri.
- CRF için kelime/offset girdisi; Transformer için tokenizer offsetleri,
  cevap maskesi ve seçilen pencere stratejisinin cevapla eşlemesi.
- Pencere stratejisini 4. kişiyle birlikte kararlaştırma; gold anotasyonlara
  bakarak kaynak seçmeme ve uzun örneklerin kapsamını raporlama.
- Veri raporu, en az üç açıklayıcı eğitim örneği, 1–2 slayt ve konuşmacı notu.

Bu kişi diğer modellerin eğitimini kendi PC'sinde yürütmez. Paylaşılan veri
kimliği değişirse 1. kişiye, model ve değerlendirme sahiplerine bildirir.

## 3. kişi — CRF baseline ve baseline deney analizi

**Kod/teknik sorumluluğu:** CRF baseline, özellikler, eğitim/tahmin komutları,
deney konfigürasyonları ve kendi modelinin açıklaması.

Teslimleri:

- Dört türü ve ortak örtüşme politikasını destekleyen CRF yaklaşımı; örneğin
  tür başına ayrı sıra etiketleyiciler. Son tasarım validation öncesi kaydedilir.
- Eğitimden öğrenilen özellikler/sözlükler; validation/test üzerinde fit yok.
- Ortak tahmin biçimine çıktı veren kod, küçük işlev testleri ve 1. kişinin
  çalıştırabileceği eksiksiz komut/konfigürasyon.
- Sınırlı validation ayar önerisi; dönen log ve sonuçların analizi, kendi
  modelindeki hataların giderilmesi ve en az üç hata örneği.
- Baseline yöntem/sonuç metni, 1–2 slayt ve konuşmacı notu.
- Bütçe uygunsa, 4. kişinin aynı Transformer kodundaki yalnız-cevap kontrolü
  için deney konfigürasyonunu ve sonuç yorumunu hazırlamak. Ortak model
  mimarisini ikinci kez yazmaz; bu deney çekirdek teslimin önüne geçmez.

Baseline'ın kaynak bağlamından hangi özelliklerle yararlandığı veya sadece
cevabı kullandığı açıkça belirtilir. Mimari ve girdi birlikte değiştiğinde
performans farkı yalnızca bağlama atfedilmez.

## 4. kişi — kaynak bağlamlı Transformer

**Kod/teknik sorumluluğu:** Küçük önceden eğitilmiş encoder, örtüşmeye uygun
token hedefi/çıktı katmanı, eğitim/tahmin kodu ve karakter aralıklarına dönüş.

Teslimleri:

- 2. kişinin ortak hazırlama çıktısıyla çalışan kaynak + cevap modeli;
  loss yalnız ilgili cevap tokenlarında hesaplanır.
- 2. kişiyle birlikte bağlam/pencereleme ve pencere tahminlerini birleştirme
  kararı. Kanıtı kaybetme, kapsama ve bellek sorunlarını açıkça raporlama.
- Ortak tahmin biçimi, küçük forward/backward veya sentetik veri kontrolü;
  1. kişinin yapacağı gerçek GPU pilotu için çalıştırma reçetesi.
- Küçük, önceden sınırlanmış validation araması; merkezi koşumdan dönen kod
  hatalarını giderme ve model sonuçlarını yorumlama.
- Bütçe varsa aynı kodda yalnız-cevap modu; 3. kişinin hazırladığı config ile
  aynı encoder, split ve eğitim bütçesinde kontrollü karşılaştırma.
- En az üç hata örneği, yöntem/sonuç metni, 1–2 slayt ve konuşmacı notu.

Bu görev 13B modelin tam eğitimini veya çok sayıda Transformer denemesini
içermez. Model, max_length ve batch gibi son seçimler GPU pilotu ve validation
sonuçlarıyla sabitlenir.

## 5. kişi — ortak değerlendirme, karşılaştırma ve sunum

**Kod/teknik sorumluluğu:** Ortak scorer, karşılaştırma tablosu, 2–3 grafik,
slaytları birleştirme ve süreli sunum.

Teslimleri:

- Bölüm bulma için character precision/recall/F1; dört tür için sınıf bazında
  support ve macro F1; sınır/tür eşleşmesinin ayrı raporlanması.
- Boş gold/tahmin, kısmi örtüşme, birden fazla tür ve offset sınırları için
  küçük sentetik scorer kontrolleri. Sonuçları beklemeden ilk gün başlanabilir.
- Kaynak scorer'ıyla aynı metrik kullanılacaksa bunun doğrulanması; aksi hâlde
  metrik proje metriği olarak adlandırılır.
- Tüm modelleri aynı manifest ve tahmin dosyalarıyla karşılaştıran tek tablo;
  örneğin sınıf F1 grafiği, model karşılaştırması ve görev türüne göre sonuç.
- Her üyenin verdiği açıklamaları/hata örneklerini toparlama; çalışmanın
  sınırlamalarını ve test öncesi protokolünü sunuma ekleme.
- Herkesin 1–2 slayt taslağını birleştirme, anlatım akışı, konuşmacı notları ve
  süreli prova. Sunumu **5. kişi** yapar; herkes kendi yöntemini açıklayabilmeli.

5. kişi bütün modelleri debug etmez, bütün hata analizlerini tek başına yazmaz
ve hazır olmayan deneyler için sonuç uydurmaz. Genel sonuçları kendi scorer'ı
ile üretir; model sahipleri kendi sonuçlarının yorumunu teslim eder.

## Ortak dosya sözleşmesi — önce kararlaştırılacak

Aşağıdaki arayüz başlangıç önerisidir; kodun var olduğu iddiası değildir.
İlk gün 2, 3, 4 ve 5. kişiler bunu birlikte sabitlemeli; 1. kişi entegrasyon
kaydını tutmalı.

### Hazırlanmış örnek

- `id`, `source_id`: string; yalnız eşleştirme/gruplama için, model girdisi değil.
- `task_type`, `generator`: rapor kırılımları; modele özellik olarak verilmez.
- `source_text`, `response`: modelin kullanabileceği girdiler. Gerekiyorsa soru
  ve talimatın serileştirme yöntemi kaydedilir.
- `labels`: `start`, `end`, `label_type` içeren liste. Karakter aralıkları
  `[start, end)` biçiminde orijinal cevaba referans verir. Örtüşmeler korunur.
- `split`: manifestle belirlenir; validation/test model fit'ine girmez.
- CRF/Transformer hazırlama ekleri orijinal offsetlerle geri eşlenebilir olmalı.

### Tahmin dosyası

Her cevap için bir JSONL satırı:

```json
{"id": "response-id", "spans": [{"start": 12, "end": 20, "label_type": "Evident Conflict"}]}
```

Sorunlu bölüm tahmin edilmiyorsa `spans: []` yazılır. Kimlik, tür ve offset
bütünlükleri scorer tarafından kontrol edilir. Hiçbir model gold etiketi
çıktısına kopyalayamaz. Tahmin üretimi yalnız model girdilerini kullanır.

### Merkezî koşum için üyeden beklenen paket

1. Kendi branch'i, PR'ı ve çalıştırılacak kod commit'i.
2. Veri manifesti ve gereken bağımlılık/sürüm bilgisi.
3. Tek çalıştırma komutu, konfigürasyon ve beklenen çıktı yolları.
4. Küçük bir işlev kontrolünün sonucu ve kodun bilinen sınırları.
5. Sonuçlar dönünce üyenin analizi, kendi hata örnekleri ve slayt taslakları.

1. kişi paketi inceleyip sıraya alır; başarısız koşum logunu ilgili model
sorumlusuna iletir. Diğer üyeler kendi bilgisayarlarında veri hazırlama,
sentetik küçük kontroller ve çıktı analizi yapabilir. **CPU baseline dahil
bütün tam train ve model tahmin koşumları 1. kişinin PC'sindedir.**

## Dosya sahipliği ve önerilen branch'ler

Dosya adları öneridir; uygulamalar henüz teslim edilmedi.

| Kişi | Önerilen kod/belge alanı | Branch |
|---|---|---|
| 1 | GPU/koşum scriptleri, ortak CLI, demo, README, protokol entegrasyonu | `feature/ragtruth-integration` |
| 2 | `data.py`, `alignment.py`, `splits.py`, veri config/raporu | `feature/ragtruth-data` |
| 3 | `baseline_crf.py`, baseline config/raporu, opsiyonel ablation config | `feature/ragtruth-baseline` |
| 4 | `transformer.py`, Transformer config/raporu | `feature/ragtruth-transformer` |
| 5 | `evaluate.py`, `plot_results.py`, karşılaştırma ve sunum | `feature/ragtruth-evaluation` |

Kendi alanındaki testler de ilgili üyeye aittir. Ortak arayüz değişikliği,
bağımlı üyelere bildirilip PR üzerinden yapılır. Bu plan başkalarının modelini
repo sahibinin veya asistanın onların adına tamamlaması için bir atama değildir.
Sunum dosyası/biçimi için yeni bir zorunluluk varsayılmaz; hoca yönergesi esas.

## Takvim ve kapsam sınırı

| Gün | Kişilerin paralel teslimi | 1. kişinin merkezî işi |
|---|---|---|
| 1 | 2 veri/split sözleşmesi; 3 CRF iskeleti; 4 model iskeleti; 5 sentetik scorer | Ortam/arayüz kontrolü ve hazır olan modelle pilot |
| 2 | 3–4 modellerin düzeltmeleri; 2 uzunluk/coverage; 5 scorer/plot iskeleti | Sıralı baseline ve Transformer train/validation koşumları |
| 3 | Herkes kendi sonuç yorumu, hata örneği ve 1–2 slayt; validation ile seçim | Sınırlı ek koşum, bütçe varsa response-only kontrolü |
| 4 | 5 karşılaştırma ve sunumu birleştirir; herkes yöntemini açıklar | Protokol commit'inden sonra tek resmî test koşumu, demo |
| 5 varsa | Tekrar üretilebilirlik kontrolü ve sunum provası | Son paket ve kayıt kontrolü |

Çekirdek teslim CRF + bağlamlı Transformer + ortak değerlendirme + küçük demo
ve sunumdur. Response-only kontrolü bütçeye bağlıdır. Her yeni model veya
geniş parametre araması, süre etkisi görülmeden plana eklenmez.

Validation kararları ve metrik kodu resmî testten önce sabitlenir. 5. kişi
final tahminler üzerinden metrikleri yeniden hesaplayabilir; bu, test sonucuna
bakarak model/eşik seçmek için kullanılmaz. İlgili protokol:
[PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md).
