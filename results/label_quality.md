# Etiket kalitesi denetimi

Kapsam: `good` kalitedeki train+validation (14.942 cevap, 12.693 span). Resmî test
etiketleri okunmadı. Hiçbir etiket değiştirilmedi veya silinmedi; bu bir temizlik
değil, ölçümdür. Güncel ham sayılar:
[2 Kasım doğrulanmış raporu](review_2026-11-02/label_quality.json).
Eski [label_quality.json](label_quality.json) ilk PR'ın tarihsel çıktısıdır.
Yeni denetim train/dev dosyalarını, kaynak gruplarını ve SHA kimliklerini
doğrular; raw response veya resmî test anotasyon dosyasını açmaz.

- **Noktalama sınırı:** Span'lerin %37,67'si noktalamayla başlıyor veya bitiyor;
  %29,84'ü cümle sonu işaretiyle bitiyor. Kelime ortası sınır yalnız 7 span'de.
  Bu, karakter F1 ile exact-span F1 arasındaki farkın olası bir parçasıdır;
  etkisi model tahminleriyle ölçülmedi.
- **Subtle Baseless Info:** Span'lerin %74,3'ü dataset metadata'sında
  `implicit_true` olarak işaretlenmiş. Bu, her bilginin gerçek dünyada ayrıca
  doğrulandığı anlamına gelmez; bu etiketler hedeflerde korunur.
- **Evident Baseless Info:** Span'lerin %25,9'u `due_to_null` işaretlidir.
- **Span uzunluğu:** Medyan 5–6 kelime, ama uçlar uzun (en fazla 239 kelime).
- **Tür notu tutarlılığı:** Açık başlığı ayrıştırılabilen 11.812 span'in
  77'sinde (%0,65) başlık `label_type` ile uyuşmuyor. 881 not ayrıştırılamadı
  (84 null, 797 tanınmayan biçim). İlk parser'ın kabul ettiği 24 şüpheli satır
  artık yorumlanmıyor. Bu otomatik başlık farkları kanıtlanmış gold hatası değildir;
  serbest metin notu ve son etiket farklı aşamalardan gelmiş olabilir.
- **İnsan uyumu ölçülemedi:** Aynı kaynakta birebir aynı metne sahip 43 çift var,
  ama hiçbirinde iki taraf da pozitif değil. Çoğu standart ret cevabı. Bu nedenle
  anotatör uyumu veya skor tavanı için kanıt sayılmaz.
- **Aynı içerikli kaynaklar:** Kaynak metni + cevap veya manifest grubu + cevap
  ile gruplayınca 44 çift var; bir çift farklı kaynak ID'lerinden geliyor.
  Bunlar da bütünüyle negatif. Exact source_id ile aynı-content grubu ayrı kavramlardır.

Boş veya eksik sınıflı küçük veri testleri de desteklenir; olmayan sınıfın
uzunluk istatistikleri null, sayıları sıfırdır. Yeni çıktı açıkça seçilir ve
üzerine yazılmaz:

```text
python scripts/audit_label_quality.py --output outputs/new-label-audit.json
```
