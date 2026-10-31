# RAGTruth — deney protokolü taslağı

> Tarihsel kayıt: Bu belge ilk plan/teslim/kontrol sırasındaki durumu anlatır.
> Güncel teknik sonuçlar [FINAL_REPORT.md](../results/FINAL_REPORT.md), nihai
> kararlar [FINAL_PROTOCOL.json](FINAL_PROTOCOL.json) içindedir. Kullanıcının
> son atamasıyla sunum dışındaki teknik tamamlama işi 1. kişi tarafından yapıldı;
> ilk uygulamaların katkı sahipliği korunur.

Durum: **TARİHSEL TASLAK; güncel nihai karar FINAL_PROTOCOL.json içindedir.** Bu dosya resmî test
değerlendirmesine izin veren donmuş bir deney protokolü değildir.

## Girdi ve hedef

- Kaynak: üretimde sağlanan içerik ve gerekli soru/talimat.
- Hedef: üretilen cevaptaki anotasyonlu karakter aralıklarını bulmak ve türünü
  tahmin etmek. Kaynak tokenlarına eğitim hedefi verilmez.
- Dört anotasyon türü ham verideki adlarıyla korunur.
- Anotasyon yorumları (`meta`), gold span metinleri, quality/implicit_true gibi
  değerlendirme sırasında bilinmeyen anotasyon alanları model girdisine
  verilmez. `source_id` ve `id` de öğrenilebilir girdi özellikleri değildir.
- `implicit_true` ve `due_to_null` korunur. Bunlar için uygulanacak hedef
  politikası eğitim verisi ve kaynak değerlendirme kodu üzerinden açıklanıp
  sabitlenmeli. Etiketler sessizce silinmemeli veya yeniden tanımlanmamalı.

## Ayrım ve sızıntı

- Resmî test aynen korunur; geliştirme ve model seçimi için kullanılmaz.
- Resmî eğitimden validation ayrılır. Aynı `source_id` grubundaki bütün
  cevaplar aynı tarafta kalmalı; tek tek cevap satırları rastgele ayrılmamalı.
- Önerilen başlangıç: seed 42, kaynak gruplarının yaklaşık %15'i validation;
  görev türlerine göre denge. Manifest ve gerçek sayılar eğitimden önce yazılır.
- Kaynak ve cevap tekrarları ayrıca raporlanır; source_id kontrolü tek başına
  farklı ID'lerdeki aynı kaynak içeriğini tespit etmez. Sorumlu üye normalize
  edilmiş kaynak tekrarlarını inceleyip gruplama kararını kayıt altına almalı.
- Resmî testte anotasyon istatistikleri incelenmez. Dosyayı ham olarak indirmek
  ve split/ID üyeliğini kontrol etmek, test performansı ölçümü değildir.

## Ön işleme

- Orijinal cevap ve karakter konumları değişmez; lowercase/strip gibi işlemler
  offsetleri sessizce bozamaz.
- İlk kontrolde 101 eğitim cevabında örtüşen anotasyonlar var. Türler farklı
  olabilir; tek etiketli BIO'ya zorlayıp anotasyon kaybetmek kabul edilmez.
  Örtüşmeleri koruyan çok etiketli/per-type hedef veya açık başka bir tasarım
  seçilmeli; tipler ve aynı tipteki sınırların geri dönüş politikası açıklanmalı.
- Uzun kaynakları kısaltma/pencereleme yalnız modele izin verilen girdilere
  dayanmalı; gold span, gold hata türü veya anotatör yorumu ile kaynak seçilmez.
- Eksik bağlam kanıtın yokluğunu taklit edebilir. Tokenizer ile kaynak/cevap
  uzunlukları, kesilen kısım ve cevap coverage'ı ölçülmeli; birincil değerlendirme
  sırasında uzun örnekler sessizce dışlanmamalı.
- Quality flag, tekrar veya başka nedenle filtreleme yapılırsa önce/sonra
  sayıları, gerekçesi ve resmî karşılaştırmaya etkisi ayrı gösterilmeli.

## Modeller ve karşılaştırma

Henüz seçilmedi. Hocanın zorunlu yöntemleri varsa plana eklenmeli.

- Temel kapsam: bir baseline, bir kaynak-bağlamlı küçük Transformer.
- Kontrollü kaynak etkisi deneyi: aynı Transformer, aynı veri ve bütçe,
  yalnız cevap girdisi. Bu varyant için bütçe pilot sonrası kararlaştırılır.
- Baseline ile bağlamlı model aynı hedef tanımı ve aynı split üzerinde ölçülür.
  Farklı mimari ve farklı girdi aynı anda değişiyorsa, fark yalnız bağlama
  atfedilemez.
- Makaledeki 13B generatif detector ile bizim token sınıflandırıcısı aynı
  mimari değildir. Yayın sonucuna karşı iddialar bu farkı açıkça belirtmeli.

## Metrikler

- Accuracy ana başarı metriği olmaz; çoğunluk sınıfı sorunlu olmayan tokenlar
  olabilir.
- Sorunlu bölümlerin birleşimi için character precision/recall/F1;
  dört tür için sınıf bazında ve macro F1; sınıf support'ları.
- Ayrı ek metrik olarak sınırları ve türü doğru olan span eşleşmeleri.
- Karakter boşluklarının, boş tahmin/boş gold'un ve kısmi örtüşmenin nasıl
  işlendiği açıkça tanımlanıp test edilir. Micro/macro ortalaması isimde yazılır.
- Orijinal makaleyle karşılaştırmadan önce kaynak scorer'ı ve `implicit_true`
  politikası kontrol edilmeli. Aynı yöntem uygulanamıyorsa sonuç açıkça proje
  metriği olarak etiketlenmeli; resmî metrik olduğu iddia edilmemeli.
- Görev türü (QA/Data2txt/Summary), üretici model ve az görülen sınıflara göre
  sonuçlar, aynı sabit tahminler üzerinden raporlanır.

## Nihai testten önce doldurulacaklar

1. Validation manifesti ve kaynak tekrarları politikası.
2. Girdi serileştirmesi, tokenizer/model revision'ları ve örtüşme kodlaması.
3. Bağlam stratejisi, max_length, batch, hassasiyet, seed ve epoch bütçesi.
4. Validation ile seçilmiş checkpoint/eşik ve seçimin gerekçesi.
5. `implicit_true`, `due_to_null`, quality ve tekrar politikaları.
6. Metrik kodu/sürümü ve anlamlı sınır durumu testleri.
7. Ortam/paket sürümleri, model ve veri checksum'ları, pilot ölçümleri.
8. Dondurulmuş protokolün commit kimliği; bundan sonra tek resmî test koşusu.

Nihai test görülünce model, eşik veya ön işleme tekrar ayarlanmaz.
