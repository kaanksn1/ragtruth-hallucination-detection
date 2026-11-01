# Teknik paketi kullanma

**V2 güncellemesi:** Güncel denetlenmiş paket `artifacts/ragtruth_v2_audited.zip`;
raporları paket içinde `results/v2/` altındadır. [V2 denetimi](../results/v2/AUDIT_TR.md)
ve [çalıştırma rehberi](V2_REPRODUCIBILITY.md) önceliklidir. Aşağıdaki V1 dosyaları
tarihsel teslimi açıklar; yeni internal holdout ile eski resmî test ayrı tutulur.

Proje, verilen kaynak ile LLM cevabını birlikte okuyup cevaptaki dört tür
halüsinasyon aralığını bulur. Kaynakta desteklenmeyen bir ifade, dünyada mutlaka
yanlış demek değildir. Etiketler veri setinin insan anotasyonlarıdır.

## Önce açılacak dosyalar

- `results/FINAL_REPORT.md`: model seçimi, validation/test sonuçları ve sınırlar.
- `results/validation_comparison.md` ve `results/test_comparison.md`: aynı
  scorer ile hesaplanan model karşılaştırmaları.
- `results/figures/validation/` ve `results/figures/test/`: karşılaştırma,
  tür bazında ve görev bazında F1 grafikleri.
- `results/error_analysis.md`: yalnız validation'dan seçilmiş hata örnekleri.
- `artifacts/comparison-demo.html`: tarayıcıda açılan, kaydedilmiş validation
  tahminlerini gösteren demo. Paket içinde adı `demo.html` olur.
- `artifacts/ragtruth_technical_handoff.zip`: demo, rapor, grafik ve kaynak kod
  paketi. ZIP içindeki `MANIFEST.json` dosyaların SHA-256 değerlerini içerir.

Bu yollar nihai paket oluşturulduğunda geçerlidir. Ham veri, model ağırlıkları
ve tam tahmin dosyaları Git'e veya hafif teslim ZIP'ine alınmaz; yerel
`data/` ve `outputs/` klasörlerinde korunur. Demo tek başına açılabilir.

## Demoyu anlatma

1. Model listesinden Transformer veya CRF seçilir.
2. Kaynak okunur; cevaptaki modelin işaretlediği aralıklar incelenir.
3. Ayrı paneldeki insan anotasyonlarıyla karşılaştırılır. Bunlar modelin çıktısı
   gibi sunulmaz.
4. Arama, görev ve hata filtreleriyle bir örneğe geçilir. Dört renk dört gerçek
   hata türünü; çizgili aralıklar örtüşen türleri gösterir.

Demo yeni bir cevap üretmez veya canlı model çalıştırmaz. Filtrelenmiş ekranın
görünümü bütün veri setinin başarı ölçümü değildir. Başarı rapordan okunur.

## Sonuçları doğru adlandırma

- **Any character F1:** Türü dikkate almadan sorunlu karakterleri bulma.
- **Typed macro character F1:** Dört gerçek türün F1 değerlerinin eşit ortalaması.
- **Exact typed span F1:** Aralığın başlangıcı, sonu ve türü birlikte doğru olmalı.

Tablodaki yüzde F1 değerleri accuracy değildir. Validation model/eşik seçiminde
kullanılmıştır; bağımsız başarı tahmini olarak sunulmaz. Nihai test öncesi
seçimler `docs/FINAL_PROTOCOL.json` ve `configs/final/` altında commit edilir.
Test sonucuna göre yeniden ayar yapılmaz. Bu proje makaledeki generatif
detector/scorer ile birebir aynı deney veya SOTA iddiası değildir.

## Bilinen sınırlar

Subtle Conflict çok az örnek içerir. Eşik seçimi ve ağırlıklı loss bu sınıfın
güvenilir biçimde çözüldüğünü tek başına göstermez. Kaynak penceresi seçimi
kanıt kaçırabilir; token sınırları tam span eşleşmesini zorlaştırabilir.
Deneyler tek seed ile yapılmıştır. CRF ile Transformer farkının tamamı yalnız
kaynak bağlamına bağlanamaz; CRF'nin kaynaklı/yalnız cevap kontrolü ayrı okunur.

Kod ve teknik paket bu teslimin kapsamındadır. Slaytları hazırlama, konuşmacı
notları ve süreli sunum provası ekibin kalan işidir. Kişisel katkılar gerçek
commit/PR geçmişine göre anlatılmalıdır.
