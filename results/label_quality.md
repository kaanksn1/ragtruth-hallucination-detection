# Etiket kalitesi denetimi

Kapsam: `good` kalitedeki train+validation (14.942 cevap, 12.693 span). Resmî test
etiketleri okunmadı. Hiçbir etiket değiştirilmedi veya silinmedi; bu bir temizlik
değil, ölçümdür. Tekrar üretim: `python scripts/audit_label_quality.py`.
Ham sayılar: [label_quality.json](label_quality.json).

- **Noktalama sınırı:** Span'lerin %37,67'si noktalamayla başlıyor veya bitiyor;
  %29,84'ü cümle sonu işaretiyle bitiyor. Kelime ortası sınır yalnız 7 span'de.
  Bu, karakter F1 ile exact-span F1 arasındaki farkın olası bir parçasıdır;
  etkisi model tahminleriyle ölçülmedi.
- **Subtle Baseless Info:** Span'lerin %74,3'ü `implicit_true`. Yani bu türün büyük
  kısmı kaynakta yazmayan ama doğru sayılan bilgi.
- **Evident Baseless Info:** Span'lerin %25,9'u `due_to_null`; bunların 1.416'sı
  Data2txt görevinde.
- **Span uzunluğu:** Medyan 5–6 kelime, ama uçlar uzun (en fazla 239 kelime).
- **Tür notu tutarlılığı:** Notu okunabilen 11.836 span'in 77'sinde (%0,65) nottaki
  tür `label_type` ile uyuşmuyor. 857 span'in notu türe çevrilemedi.
- **İnsan uyumu ölçülemedi:** Aynı kaynakta birebir aynı metne sahip 43 çift var,
  ama hiçbirinde iki taraf da pozitif değil. Çoğu standart ret cevabı. Bu nedenle
  anotatör uyumu veya skor tavanı için kanıt sayılmaz.
