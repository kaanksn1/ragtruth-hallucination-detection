# Serdar Kaan Kesen — 220911810

GitHub kullanıcı adı: [kaanksn1](https://github.com/kaanksn1)

Sorumluluk: 1. kişi — merkezî koşum, entegrasyon ve demo; 31 Ekim'de açıkça atanan
sunum dışı teknik tamamlama işi. Grup: Anonymous.

> Bu dosya commit ve PR kayıtlarından derlendi; üye tarafından doğrulanmalıdır.

## Tamamladığım işler

- Proje kapsamını, ekip iş akışını, GPU ön kontrolünü, SHA-256 doğrulamalı veri
  indirmeyi ve `scripts/run_job.py` ile sıralı, kayıtlı merkezî koşum altyapısını kurdum.
- 2. kişinin veri PR'ında dört tür hedefin batch dönüşümünü düzelttim ve cevap
  dışı loss maskesini korudum (`3854f3f`).
- CRF entegrasyonunda veri ve checkpoint kimliği doğrulamasını, Windows'ta ASCII dışı
  model yolu desteğini ve CI'daki sabit CRF kurulumunu ekledim.
- Transformer entegrasyonunda checkpoint veri kimliği ve donmuş test girdileri
  kontrollerini ekledim. Gerçek GPU pilotunu ve uzunluk profilini kaydettim.
- Tam CRF ve Transformer koşumlarını yürüttüm. Ortak değerlendirmeyi, doğrulama
  verisine dayalı eşik kalibrasyonunu, önceden beyan edilmiş ağırlıklı BCE denemesini
  ve çevrimdışı demoyu hazırladım.
- Bilimsel kararları, tahmin işlerini ve değerlendirme protokolünü sırasıyla ayrı
  commit'lerle dondurup resmî testi çalıştırdım. Sonuçları, grafikleri, Türkçe
  bulguları ve teknik devri yayımladım.
- Temiz klonda donmuş config checksum'larını koruyan satır sonu düzeltmesini yaptım.
- `feature/ragtruth-v2` branch'inde focal, BCE ve yapılandırılmış V2 deneylerini
  ve iç holdout çalışmasını yaptım. Bu branch `main`'e birleşmedi; sunum V1 sonuçlarını kullanır.

Ekip arkadaşlarının PR'larını inceleyip birleştirdim. Veri, CRF ve Transformer'ın ilk
tasarım ve uygulamaları ilgili üyelerin katkısıdır. Ortak değerlendirme başlangıçta
5. kişiye planlanmıştı, ancak 31 Ekim atamasıyla tarafımdan tamamlandı.

## Doğrulama ve deneyler

- Çalıştırma komutları: `python -m unittest discover -s tests -v` ve `docs/RUNNING.md`
- Veri bölümü: resmî testte 2.700 cevabın 2.675 `quality=good` cevabı.
- Çıktılar: `results/FINAL_REPORT.md`, `results/BULGULAR_TR.md`, `results/test_comparison.md`
- PR #6 kaydı: CRF ortamında 193 senaryonun 162'si geçti, 31'i atlandı;
  Transformers ortamında 190'ı geçti, 3'ü atlandı.
- Resmî test sonuçları:

| Model | Any karakter F1 | Dört tür macro F1 | Exact typed-span F1 |
|---|---:|---:|---:|
| Kaynak özellikli CRF | %35,44 | %19,76 | %6,67 |
| Yalnız cevap CRF | %20,74 | %11,95 | %6,15 |
| Kalibre ModernBERT | %52,06 | %29,84 | %6,78 |

- Sınırlamalar: tek seed; Subtle Conflict F1 üç modelde de 0; exact sınır başarısı düşük.

## GitHub kanıtı

- [Proje başlangıcı ve GPU ön kontrolü](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/541fdac8633debdb68e87846836bb4fa5b0118eb)
- [Batch hedef düzeltmesi](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/3854f3fe1f2dd983e3fd14c0082c01e0a891e25d)
- [Ortak değerlendirme, kalibrasyon ve demo](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/8f9b57a652718c947d9858c15880bfda979c0186)
- [Bilimsel kararların dondurulması](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/ce4c1341c819efcb5f8bf9171e4ec8094d2bcb74)
- [Test sonuçlarının yayımlanması](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/8a6a9f6d611f1b704e045df7a56dc7dbb0f429eb)
- PR'larım: [#1](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/1),
  [#5](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/5),
  [#6](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/6),
  [#7](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/7)
- [V2 branch'i](https://github.com/kaanksn1/ragtruth-hallucination-detection/tree/feature/ragtruth-v2)

## Sunuma katkım

- Sunumda kullanılan resmî test sonuçlarını, grafikleri, bulguları ve teknik devri
  hazırladım. Slaytlar ve sunum bu atamanın dışındadır.
