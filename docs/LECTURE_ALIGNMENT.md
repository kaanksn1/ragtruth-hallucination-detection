# RAGTruth planının ders slaytlarıyla eşleştirilmesi

İnceleme tarihi: 29 Ekim 2026. Kullanıcının paylaştığı beş dosya toplam 175
slayttır. Slayt numaraları dosyadaki gerçek sunum sırasını izler. Metin ve
konuşmacı notları okundu; Week 1 ve W02'de görsel olarak saklanan içerikler
ayrıca incelendi. Kaynak PPTX dosyaları değiştirilmedi veya projeye kopyalanmadı.

## Sonuç ve kanıt sınırı

**Mevcut CRF baseline + kaynak bağlamlı küçük Transformer planı dersin yöntem
içeriğiyle uyumludur.** Bu beş dosya, P2'de bütün model ailelerinin birlikte
uygulanmasını açıkça zorunlu tutmuyor. Bununla birlikte slaytlardan, iki modelin
hoca tarafından kesin kabul edildiği sonucu da çıkarılamaz. Kullanıcının
paylaştığı güncel P2 duyurusu ve hocanın ayrıca vereceği proje yönergesi esas
alınmalı; yöntem sayısı ve P2 sunum süresi henüz kesinleşmiş sayılmamalı.

RAGTruth'ta dört hata türünün karakter aralıklarını tahmin etmek, genel token
sınıflandırma mekanizmasının bu veri setine uyarlamasıdır. Slaytlardaki NER
etiketlerini veya tek etiketli örnek kodları aynen kopyalamak yeterli değildir.

## Dosya ve slayt eşleştirmesi

| Kaynak dosya | Slayt | Gözlenen içerik | RAGTruth'a etkisi |
|---|---|---|---|
| `8_POSNER_intro_May_6_2021.pptx` | 5–6, 12–13, 17, 20–25 | Hizalı token etiketleri, özellikler, bölüm sınırları, BIO/IO/BIOES, HMM/CRF/neural alternatifleri | Token hedeflerini ve gerçek tür/sınır ayrımını açıklamak; manuel özellikli CRF için ders dayanağı |
| `week6-pro-standalone-lecture.pptx` | 16–19, 23, 24–42 | Alt kelime hizalama, maske, span F1, HMM/Viterbi, CRF, geçişler, neural token başlıkları | Hizalama ve loss maskesi; CRF skor/decoder açıklaması; tam pipeline değerlendirmesi |
| `week3-pro-standalone-lecture.pptx` | 4–9, 10–31, 32–36 | Klasik metin modelleri, embedding/CNN/RNN/BERT, padding, truncation ve model karşılaştırması | Kaynak/cevap bağlamı ve uzunluk kararı, aynı splitte karşılaştırma, validation ile seçim, süre/hata raporu |
| `COE025 - Natural Language Processing - W02 (1).pptx` | 14–25, 26–27 | NB/LR/SVM ve özellikler; açıkça Project 1 gereklilikleri/rubriği | Genel baseline ve değerlendirme ilkeleri yararlı; P1 model zorunluluğu ve süre sınırı P2'ye otomatik taşınmaz |
| `week1-presentation (1).pptx` | 12, 14–24, 30–34 | Ders akışı, token/subword/temsil, dört proje; P2 için HMM/CRF/BiLSTM-CRF ve Viterbi öğrenme hedefleri | P2'nin sequence labeling bağlamı; öğrenme hedefi listesini bütün modeller için uygulama zorunluluğu gibi okumamak |

### P1 ve P2 ayrımı

W02, slayt 26'nın yerel başlığı `Project 1: Text Classification`:

- Beş öğrenci, tek metin sınıflandırma problemi.
- Naive Bayes ve dersten başka en az bir yöntem.
- Benchmark karşılaştırması.
- Week 4 için üç dakika sunum ve iki dakika soru-cevap.
- Sunum öncesi GitHub deposu.

Slayt 27'deki teknik/özgünlük/sunum yüzdeleri de aynı P1 bölümündedir. Bu
koşullar P2'nin güncel zorunlu model listesi, sunum süresi veya puan dağılımı
olarak kaydedilmedi. Week 1 slayt 31 P2'yi NER/POS üzerinden tanıtır; güncel P2
duyurusu ise daha farklı token sınıflandırma görevlerine açıkça izin veriyor.

### Yöntemler ve Viterbi

- POS/NER giriş slaytları 13 ve 25'te yöntem aileleri listelenir. Denetimli
  öğrenmede elle etiketli eğitim verisi gerektiği, bütün yöntemlerin projede
  zorunlu olduğu anlamına gelmez.
- Week 6 slayt 23 HMM, CRF ve neural tagger'ı alternatif modelleme yolları
  olarak açıklar; bunlar tek modelin zorunlu ardışık aşamaları değildir.
- Week 6 slayt 32–38 CRF'nin gözlem ve geçiş skorlarını, normalizasyonunu,
  öğrenmesini ve Viterbi ile en iyi dizinin seçimini ele alır.
- Week 6 slayt 39, öğrenilmiş sonlu geçiş skorları ile sert BIO kısıtlarının
  farkını gösterir. Salt CRF kullanmak geçerli BIO dizisi garantisi sayılmaz.
- Week 6 slayt 40–41, bağlamsal encoder çıktısına yerel başlık veya CRF
  decoder eklenebileceğini anlatır. Transformer+CRF otomatik olarak üstün bir
  model gibi sunulmamalı.

## Görevlere eklenen açıklama ve teslim ayrıntıları

Mevcut kişi/branch sahipliği ve tüm eğitimlerin sahibin PC'sinde çalışması
kuralı değişmez. Her üye aşağıdaki ayrıntıları kendi tesliminde açıklar:

| Kişi | Açıklayacağı / doğrulayacağı konu |
|---|---|
| 1 — entegrasyon ve koşum | Sabit veri/model sürümleri, aynı split, koşum süresi ve GPU belleği, validation ile seçim; modellerin kullandığı girdilerin açık kaydı |
| 2 — veri | Kelime/alt kelime/karakter eşlemesi; span sınırları; kaynak, cevap ve padding maskeleri; pencereleme coverage'ı |
| 3 — CRF | Özelliklerin ne olduğu, unary/geçiş skorları, sequence decoding/Viterbi; kullanılan BIO kısıt politikası ve dört tür/örtüşme tasarımı |
| 4 — Transformer | Her cevap tokenı için çıktı; bağlamsal encoder ile çıktı decoder'ının farkı; sadece cevap konumlarında loss; kaybolan kanıt ve pencere tahminlerinin birleştirilmesi |
| 5 — değerlendirme ve sunum | Precision/recall/F1, sınıf desteği ve macro F1; karakter örtüşmesi ile exact boundary+type F1 ayrımı; sınır ve tür hatalarının örnekleri |

Her üye 1–2 slayt ve konuşmacı notunu kendi bölümüne göre hazırlar. Sunumu
5. kişi yapar. Bu eşleştirme yeni bir model uygulaması tamamlandığı anlamına
gelmez; ilgili teknik teslimler hâlen sahiplerine aittir.

## Uygulamada korunacak ayrımlar

1. **Kelime ve model tokenı:** WordPiece gibi tokenizer'lar kelimeyi bölebilir.
   İlk alt kelimeyi veya bütün alt kelimeleri etiketleme kararı tutarlı olmalı.
   Week 6 slayt 16/42 bu ayrımı açıkça işler.
2. **O ve loss dışlama:** `O` gerçek bir dışarıda etiketi olabilir. Özel/kaynak/
   padding konumlarını loss dışında tutmak, bunları `O` yapmakla aynı işlem
   değildir. Kayıp fonksiyonuna göre ignore index veya geçerli konum maskesi
   kullanılır; `-100` her loss fonksiyonunda otomatik anlam taşır diye varsayılmaz.
3. **Örtüşen anotasyonlar:** RAGTruth'un 101 eğitim cevabında örtüşme saptandı.
   Dört tür için tek softmax/BIO dizisine zorlama etiket kaybına yol açabilir.
   Tür başına başlıklar/diziler gibi tasarımlar veri sözleşmesiyle birlikte
   değerlendirilir. Aynı tipte örtüşen sınırların politikası da açık olmalı.
4. **Dört tür ve kodlama sayısı:** BIO kodlaması gerçek hata türlerini artırmaz.
   Dersin düz NER örneğindeki `2n+1` etiket sayısı, dört tür için tek dizinin
   bütün RAGTruth anotasyonlarını temsil ettiğinin kanıtı değildir.
5. **Token ve span başarısı:** Accuracy ana hedef değildir. Karakter örtüşmesi,
   bölüm sınırının tam eşleşmesi ve türün doğru olması ayrı ölçümlerdir.
   Orijinal RAGTruth metriği ile proje metriği eşitlenmeden benchmark iddiası
   kurulmaz.
6. **Metin ve token sınıflandırma:** Week 3 slayt 18'de global pooling bütün
   metne tek çıktı üretir; slayt 31'in BERT örneği de metin sınıflandırır.
   RAGTruth'ta token konumları ve orijinal karakter karşılıkları korunmalıdır.
7. **Bağlam ve decoding:** Kaynağı okumak ile komşu çıktı etiketlerini birlikte
   seçmek farklı tasarım kararlarıdır. Aynı encoder'ın response-only kontrolü
   kaynak katkısını ölçebilir; farklı mimarili CRF/Transformer farkı tek başına
   yalnız bağlamın etkisi olarak yorumlanmaz.

## Açık kalan proje koşulları

- Hocanın bu veri seti/göreve ilişkin yanıtı ve P2 tablosunun güncel kontenjanı.
- P2 için zorunlu model aileleri ve minimum yöntem sayısı.
- P2 sunum süresi ve güncel teslim/rubrik ayrıntıları.

Bu belgedeki eşleştirme, bu soruların hocadan alınmış cevabı değildir.

## İncelenen dosyaların kimliği

Kaynak dosyalar kullanıcı tarafından Downloads klasöründe sağlandı. Aşağıdaki
SHA-256 değerleri incelenen yerel dosyaları tanımlar; bağımsız yayıncı
checksum'ı oldukları iddia edilmez.

- `8_POSNER_intro_May_6_2021.pptx` (26 slayt): `7414bffcc6fa055e8df8f10e2a03753b671b86a8fc6ea53a794d107fdab5e31a`
- `week6-pro-standalone-lecture.pptx` (44 slayt): `ddeb5f5e0b347fec27b46e9d0abac3f9cac082f629f06d154e6f5e53fab51381`
- `week3-pro-standalone-lecture.pptx` (36 slayt): `4028e546f54ad256db4f92f3f905cd9742a11e96e8c12f49562b287a1eed96ad`
- `COE025 - Natural Language Processing - W02 (1).pptx` (32 slayt): `f54adf1be2cbd232caefdc872c9e87945fbb1d1cb3fbe01bf3df8117938b0124`
- `week1-presentation (1).pptx` (37 slayt): `fd94796fc7ece9b8244190d551d3de308fe4d2c2b7a8316985472ce06f5cbc96`

İlgili planlar: [TEAM.md](TEAM.md), [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md).
