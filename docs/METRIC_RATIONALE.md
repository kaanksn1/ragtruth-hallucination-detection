# Metrik seçiminin sınıf dengesizliğiyle gerekçesi

Tarih: **5 Kasım 2026**. Bu belge, değerlendirme geri bildirimi sonrasında
gerekçeyi görünür biçimde genişletir. Mevcut scorer, donmuş model seçimi,
eşikler, split'ler veya tarihsel resmî test skorları değiştirilmez.

## Veride iki ayrı dengesizlik var

[Eğitim/validation veri raporu](DATA_REPORT.md), değerlendirmeye dahil olan
`quality=good` cevaplar için şu sayıları verir:

| Eğitim desteği | Sayı/oran |
|---|---:|
| Herhangi bir anotasyonla örtüşen kelime tokenı | yaklaşık %5,6 |
| Evident Conflict anotasyonu | 4.017 |
| Subtle Conflict anotasyonu | 160 |
| Evident Baseless Info anotasyonu | 4.654 |
| Subtle Baseless Info anotasyonu | 2.006 |

Birinci dengesizlik, sorunlu ve sorunsuz konumlar arasındadır. Tüm kelimeleri
`O` olarak sınıflandırmak, rapordaki yuvarlatılmış kelime oranına göre yaklaşık
%94,4 ikili token accuracy verebilir; buna karşın hiçbir pozitif konumu bulmaz
ve pozitif sınıf F1'ı sıfır olur. Bu sayı **açıklayıcı çoğunluk örneğidir**;
eğitilmiş bir baseline skoru, karakter accuracy'si veya test sonucu değildir.

İkinci dengesizlik, dört hata türünün kendi arasındadır. Örneğin eğitimde
Subtle Conflict anotasyon sayısı 160 iken Evident Baseless Info 4.654'tür.
Anotasyon sayısı ile karakter desteği aynı şey değildir; her ikisinin de
raporlanması gerekir. Bu dağılımlar testten model seçmek için kullanılmaz.

## Birincil ölçüt: dört türün character macro F1'ı

Her tür için karakter konumları üzerinden `TP`, `FP`, `FN` hesaplanır:

`F1_t = 2 TP_t / (2 TP_t + FP_t + FN_t)`

`typed_macro_F1 = (F1_EC + F1_SC + F1_EBI + F1_SBI) / 4`

Dört tür eşit ağırlık alır; sık görülen türdeki çok sayıdaki karakter, az
görülen türün tamamen başarısız olmasını tek bir toplam içinde gizleyemez.
Payda sıfırsa ilgili F1 sıfırdır; o tür ortalamadan çıkarılmaz. Farklı türlerin
aynı karakterde örtüşmesi korunur, tek softmax sınıfına zorlanmaz.

Bu, [FINAL_PROTOCOL.json](FINAL_PROTOCOL.json) içindeki **tarihsel V1 Transformer
seçiminin** birincil ölçütüdür. Seçim kuralı: validation typed macro F1,
eşitlikte any-character F1; aileler tam eşitse temel/ağırlıksız model.
Geri bildirim nedeniyle resmî test görüldükten sonra bu kural değiştirilmez.

Tarihsel CRF hiperparametre/decoder taramasında seçim ölçütü any-character
F1'dı; [CRF raporu](BASELINE_CRF.md) bu politikayı açıklar. Pozitif bölge F1'ı
sorunsuz token çoğunluğuyla accuracy gibi yükselmez, fakat nadir hata türlerini
eşit ağırlıklandırmaz. Bu nedenle CRF için de tür bazında ve macro F1 ayrıca
gösterilir. Yeni RNN/CNN/BERT ekleri macro ardından any F1 kullanır; sabit
ayarlı HMM için validation hiperparametre araması yapılmaz. Model aileleri
arasında aynı seçim/arama bütçesi kullanılmış gibi bir iddia kurulmaz.

Macro F1'ın bedeli vardır: az örnekli sınıfta birkaç tahmin büyük oynama
yaratabilir. Dört sınıfa eşit ağırlık vermek, kullanım sıklığıyla ağırlık
vermekle aynı hedef değildir. Bu yüzden macro F1 tek başına sunulmaz.

## İkincil ölçüt: any-character micro precision/recall/F1

Her cevapta dört türün kapladığı karakterlerin birleşimi alınır. Cevaplar
üzerinden TP/FP/FN toplanır; cevap F1'larının ortalaması alınmaz. Bu ölçüt,
türden bağımsız olarak hatalı bölgenin bulunmasını değerlendirir; farklı türdeki
doğru konum tahmini any-F1'a katkı yapabilir ama typed F1'a yapmaz.

Pozitif precision, gereksiz yere işaretlenen doğru metinlerin; recall ise
kaçırılan hatalı metinlerin maliyetini görünür kılar. F1 bu ikisini birlikte
izler ve büyük sayıdaki true negative karakterle yapay olarak yükselmez.
Micro toplam uzun cevaplara ve uzun span'lere daha fazla ağırlık verebildiği
için tür bazında support, F1 ve görev/üretici kırılımlarıyla birlikte okunur.

Sonraki any-F1 öncelikli deneyler **ayrı bir seçim hedefidir**. Bunlar tarihsel
macro öncelikli çalışmanın aynısı olarak sunulmaz; modlar
[CALIBRATION.md](CALIBRATION.md) içinde ayrılır.

## Ek ölçüt: exact typed-span F1

Başlangıç, bitiş ve türün üçü de doğru olmalıdır. Yakın ama yanlış sınırlar
kısmi kredi almaz. Character F1'ın bölgesel örtüşmede verdiği krediyi
tamamlar; sınır doğruluğunu ayrı sınar. Ham aynı-tür örtüşmeleri bu metrik için
birleştirilmez; yalnız birebir aynı span tekrarları tek sayılır.

Cevapta en az bir hata var/yok üzerinden hesaplanan **response-level F1**
ayrıca açıklayıcıdır; hatanın yerini veya türünü ölçmez ve V1 model seçiminde
kullanılmamıştır.

## Sonuçların sınırı ve yeniden üretim

V1 resmî testte Subtle Conflict yalnız 15 pozitif cevapta bulunur ve üç
modelin bu türdeki F1'ı sıfırdır. Bu başarısızlık macro ortalamasına dahil
edilir. Any-F1 yüksek diye nadir türlerin çözüldüğü iddia edilmez; tek seed
sonuçları güvenilirlik/istatistiksel üstünlük kanıtı olarak sunulmaz.

Koordinatlar orijinal cevaptaki Unicode karakterleridir, boşluklar dahildir.
Ham metin ve anotasyon konumları normalize edilmez; `quality=good` filtresi ve
aynı cevap kümesi bütün model karşılaştırmalarında korunur. Tam tanımlar,
boş-payda politikası ve çalıştırma örnekleri [EVALUATION.md](EVALUATION.md)
içindedir; sınıf destekleri [veri raporunda](DATA_REPORT.md) ve
[nihai teknik raporda](../results/FINAL_REPORT.md) bulunur.

Gerekçenin daha kısa tarihsel hali, teslim öncesi
[PROTOCOL_DRAFT.md / Metrikler](PROTOCOL_DRAFT.md#metrikler) bölümünde
çoğunluk tokenları nedeniyle accuracy'yi ana metrik yapmama kararı olarak
yer alır. Bu yeni açıklama, önceki sunumun aynı ayrıntıları içerdiği iddiası
değildir.
