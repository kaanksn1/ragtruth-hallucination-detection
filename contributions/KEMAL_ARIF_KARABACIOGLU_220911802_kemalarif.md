# Kemal Arif Karabacıoğlu — 220911802

GitHub kullanıcı adı: kemalarif
Sorumluluk: 3. kişi — CRF baseline, config'ler ve koşum komutları

## Tamamladığım işler

- `scripts/baseline_crf.py`: dört anotasyon türünün her biri için ayrı linear-chain CRF
  kuran baseline'ı hazırladım. Ayrı modeller, farklı türdeki örtüşen anotasyonların
  kaybolmasını önler. Kaynak özellikleri ve posterior çözümleme bu dosyadadır.
- Modeli CRFsuite (`python-crfsuite==0.9.12`), L-BFGS ve L1/L2 düzenlileştirme ile
  kurdum; tahminler ortak `{id, spans}` biçimindedir.
- `configs/crf.json`, `configs/jobs/crf_smoke.json`, `configs/jobs/crf_validation.json`,
  `requirements-crf.txt`: config ve merkezî koşum işlerini ekledim.
- `tests/test_baseline_crf.py`: CRF baseline testlerini ekledim.
- `docs/BASELINE_CRF.md`: yöntemi ve çalıştırma adımlarını belgeledim.

Veri kimliği doğrulaması, Unicode model yolu desteği ve CI kurulumu repo sahibinin ayrı
commit'leridir. Tam eğitim merkezî olarak 1. kişinin PC'sinde çalıştırıldı.

## Doğrulama ve deneyler

- Çalıştırma komutu: `python scripts/run_job.py configs/jobs/crf_validation.json`
- Kullanılan veri: `quality=good` 12.702 train ve 2.240 validation cevabı.
- Tam test sonuçları ortak değerlendirmeden gelir: kaynak özellikli CRF any F1 %35,44,
  macro F1 %19,76; yalnız cevap CRF any F1 %20,74, macro F1 %11,95.
- Bulgu: kaynak özellikleri yalnız cevap modeline göre sonuçları belirgin artırıyor.
- Sınırlamalar: Subtle Conflict F1 iki CRF'te de 0; sınırlı iterasyon nedeniyle yakınsama
  iddia edilmez; karşılaştırma tek seed'dir.

## GitHub kanıtı

- Kendi commit bağlantılarım (`feature/ragtruth-baseline`):
  - [Tür başına CRF, kaynak özellikleri ve posterior çözümleme](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/b402982d47d6896ece68b3b7c4bb133b43870e2f)
  - [Birim testleri](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/3d8819ccb10cf3ef82f3c37a97f49c9778975eef)
  - [Config ve merkezî koşum işleri](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/b40e08bd8fd7c14c65c4c83ed88db61f9b07f6e0)
  - [Yöntem ve çalıştırma belgesi](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/db642bf2a8cbe6b35c01d15eef09028f83515d88)
- Kendi PR bağlantım: [#3 CRF baseline](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/3)

## Sunuma katkım

- CRF yöntemi `docs/BASELINE_CRF.md` içinde hazır. Repoda bu üyeye ait ayrı bir slayt
  veya hata analizi kaydı yoktur.
