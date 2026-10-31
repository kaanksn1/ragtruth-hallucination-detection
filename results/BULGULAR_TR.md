# Teknik sonuçların Türkçe özeti

Kaynak metin ve LLM cevabını okuyup, cevaptaki kaynakla çelişen veya kaynakta
desteklenmeyen aralıkları buluyoruz. Dört insan anotasyonu türünü koruyoruz.
Bu, genel amaçlı doğruluk garantisi veren bir sistem değildir.

## Nihai sonuçlar

Resmî testte `quality=good` olan **2.675 cevap** aynı scorer ile değerlendirildi.
Modeller, eşikler ve veri/kod kimlikleri testten önce commit ile sabitlendi.
Aşağıdaki değerler **F1 yüzdesidir; accuracy değildir.**

| Model | Hatalı karakterleri bulma F1 | Dört türün macro F1'ı | Tam sınır + tür eşleşmesi F1 |
|---|---:|---:|---:|
| Kaynaklı CRF | 35,44 | 19,76 | 6,67 |
| Yalnız cevaplı CRF | 20,74 | 11,95 | 6,15 |
| Seçilen Transformer | 52,06 | 29,84 | 6,78 |

Transformer toplam karakter bulma ölçümünde en iyi sonucu veriyor. Bununla
birlikte tam aralık sınırlarını ve türü birlikte doğru bulma başarısı düşük.
Bu iki ölçümü birbirinin yerine kullanmamalıyız.

Kaynaklı ve yalnız cevaplı CRF aynı seçilmiş düzenlileştirme/decoder ayarını
kullanır. Kaynak özelliklerini eklemek bu deneyde karakter F1'ını **14,70 yüzde
puan** artırdı. CRF–Transformer farkında ise mimari de değiştiği için bütün
farkı yalnız kaynak bağlamına bağlayamayız.

## Validation'da ne seçtik?

İlk Transformer üç epoch eğitildi. Kaydedilen iki checkpoint ve önceden ilan
edilen 11 eşik, yalnız validation üzerinde karşılaştırıldı. Epoch 2 ve tür
başına `0.15 / 0.01 / 0.3 / 0.05` eşikleri seçildi; sıra Evident Conflict,
Subtle Conflict, Evident Baseless Info, Subtle Baseless Info şeklindedir.

Sabit `0.5` eşiğine kıyasla validation'da karakter bulma F1'ı **44,64 → 54,05**,
typed macro F1 **22,44 → 32,01** oldu. Tam sınır + tür F1'ı ise
**11,92 → 7,00** düştü. Eşik seçiminin hedefi dört türün karakter macro F1'ıydı;
bu işlem bütün metrikleri aynı anda iyileştirmiyor. Validation bu seçimlerde
kullanıldığından bağımsız başarı tahmini değildir.

Az görülen türler için bir ek eğitim denendi: pozitif loss ağırlıkları yalnız
train cevap tokenlarından hesaplandı, karekök kuralı ve 50 tavanı kullanıldı.
Bu denemenin ilan edilmiş 15 eşikli aramasında validation macro F1 **29,42**
çıktı. İlk kalibre edilmiş modelin **32,01** değerini geçemediği için seçilmedi
ve ek model resmî testte çalıştırılmadı. İki denemenin eşik gridleri de farklı
olduğundan farkın tamamını yalnız loss ağırlığına bağlamıyoruz.

## Açık kalan model sınırlamaları

- **Subtle Conflict test F1'ı üç modelde de sıfır.** Bu türün train'de 160,
  validation'da 25 anotasyon aralığı var; testte 15 cevapta bulunuyor. Eşik
  düşürmek veya ağırlıklı loss kullanmak bu sınıfı güvenilir biçimde çözmedi.
- Transformer, Subtle Baseless Info türünde %17,40 F1 aldı; kaynaklı CRF'nin
  %18,71 değerinin altında kaldı. Toplam metrikteki üstünlük her tür için geçerli değil.
- Uzun bağlamda eğitim için seçilen kaynak penceresi kanıtı kaçırabilir.
  Token ve boşluk sınırları tam span eşleşmesini zorlaştırabilir.
- Deneyler tek seed ile yapıldı; çoklu seed veya istatistiksel güven aralığı
  sonucu iddia etmiyoruz. Testten sonra model/eşik ayarı yapılmadı.

İnsan etiketlerini biz üretmedik. Katkımızı gerçek veri bütünlüğü kontrolleri,
kaynak gruplu ayrım, örtüşen etiketlerin korunması, modeller, sınırlı deneyler,
ortak değerlendirme ve hata analizi üzerinden anlatmalıyız. BIO kodlaması dört
gerçek hata türünü daha fazla anlamsal sınıfa dönüştürmez.

Ayrıntılar: [teknik rapor](FINAL_REPORT.md), [test tablosu](test_comparison.md),
[validation hata örnekleri](error_analysis.md), [teslim rehberi](../docs/TECHNICAL_HANDOFF.md).
