# Samed Ortakaya — 210901024

GitHub kullanıcı adı: SmtOrtakaya
Sorumluluk: 2. kişi — veri sözleşmesi, kaynak bazlı bölümleme ve hizalama

> Bu dosya commit ve PR kayıtlarından derlendi; üye tarafından doğrulanmalıdır.

## Tamamladığım işler

- `ragtruth_data/`: ortak veri şeması, yükleme, bölümleme, hizalama, özellik ve analiz
  modüllerini hazırladım.
- `docs/DATA_FORMAT.md`: orijinal metni, `[start, end)` offset'lerini, dört türü, örtüşen
  anotasyonları ve anotasyon metadata'sını koruyan ortak biçimi tanımladım. Model girdisi
  yalnız `source_text` ve `response` alanlarıdır.
- `data/splits/split_manifest_v1.csv`, `configs/data.json`: seed 42 ile %15 validation
  ayıran, görev türü ve kaynak veri setine göre tabakalı, kaynak grubu bazlı manifesti
  hazırladım. Aynı içerikli farklı kaynak ID'leri tek grupta tutuldu.
- `ragtruth_data/alignment.py`: CRF için kelime ve tür başına O/B/I etiketlerini,
  Transformer için cevap ve kaynak pencerelerini, pencereden cevaba eşlemeyi hazırladım.
- `docs/DATA_REPORT.md`, `scripts/prepare_data.py`: kalite işaretleri, tekrarlar, dört tür,
  örtüşmeler, uzunluklar ve açıklayıcı örnekleri raporladım.
- `docs/DATA_WINDOWING.md`, `scripts/window_report.py`: pencere ölçümlerini yaptım, pencere
  örtüşmesine sınır ekledim ve ModernBERT-base ile 2048 token önerisini belgeledim.
- `tests/test_data_alignment.py`, `tests/test_data_splits.py`, `tests/fake_tokenizer.py`:
  hizalama ve bölümleme testlerini ekledim.

Dört tür hedefin batch dönüşümündeki düzeltme repo sahibinin ayrı commit'idir.

## Doğrulama ve deneyler

- Çalıştırma komutları: `python scripts/prepare_data.py` ve
  `python scripts/window_report.py --tokenizer answerdotai/ModernBERT-base --max-length 512 1024 2048`
- Kullanılan veri: train 2.138 kaynak / 12.828 cevap, validation 377 / 2.262.
  Resmî test 450 / 2.700 olarak ayrıldı; PR kaydına göre anotasyonları okunmadı.
- Sonuç: iki tarafa bölünen kaynak grubu 0; PR kaydına göre 12.756 anotasyonun hepsinde
  offset metinle eşleşti ve 45 test geçti.
- Bulgular: `quality=good` verisinde Subtle Conflict train'de 160, validation'da 25
  anotasyon. 2048 token ile cevapların yaklaşık %99,1'i kaynağıyla tek parçaya sığıyor.
- Sınırlama: ilk uzunluk ölçümleri yaklaşık tokenizer ile yapıldı; pencere stratejisi
  sonraki ModernBERT ölçümüyle güncellendi.

## GitHub kanıtı

- Kendi commit bağlantılarım (`feature/ragtruth-data`):
  - [Veri sözleşmesi, gruplu bölümleme ve hizalama](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/404097b1f543c3ecf5dcfbd2b2085a2d76ff84d7)
  - [Pencere örtüşme sınırı ve ModernBERT ölçümleri](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/14f6f5279c05d6f7f7b67ad80ab9677d14349163)
- Kendi PR bağlantım: [#2 Ortak veri sözleşmesi, manifest ve hizalama](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/2)

## Sunuma katkım

- Veri istatistikleri ve açıklayıcı örnekler `docs/DATA_REPORT.md` içinde hazır. Repoda
  bu üyeye ait ayrı bir slayt kaydı yoktur.
