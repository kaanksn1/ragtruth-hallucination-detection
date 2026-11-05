# Teslim edilen veri ve kökeni

5 Kasım 2026 güncellemesiyle ham ve işlenmiş veri dosyaları Git teslimine
dahil edilir. Önceki sürümlerde bu iki dizin `.gitignore` ile dışarıdaydı;
indirme ve hazırlama scriptleri mevcut olsa da dosyaların kendileri yoktu.

Kaynak: [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth),
sabit revision `c103204b9ce28d6bbad859304bf30de72b8ed8fe`.
Kaynak adresleri, revision ve ham SHA-256 değerleri
[`configs/source.json`](../configs/source.json) içindedir. RAGTruth hazır,
insan tarafından etiketlenmiş veridir; bu proje yeni corpus veya yeni insan
anotasyonları üretmemiştir.

## Dosyalar

| Yol | İçerik |
|---|---|
| [`raw/response.jsonl`](raw/response.jsonl) | 17.790 orijinal LLM cevabı ve upstream anotasyonları; byte'ları değiştirilmemiştir |
| [`raw/source_info.jsonl`](raw/source_info.jsonl) | 2.965 orijinal kaynak ve üretim prompt'u; byte'ları değiştirilmemiştir |
| [`raw/LICENSE`](raw/LICENSE) | Particle Media'nın upstream MIT bildirimi |
| [`processed/train.jsonl`](processed/train.jsonl) | 12.828 hazırlanmış eğitim kaydı; 12.702'si `quality=good` |
| [`processed/validation.jsonl`](processed/validation.jsonl) | 2.262 validation kaydı; 2.240'ı `quality=good` |
| [`processed/test.jsonl`](processed/test.jsonl) | 2.700 resmî test kaydı; V1 değerlendirmesinde 2.675'i `quality=good` |
| [`processed/crf_train.jsonl`](processed/crf_train.jsonl) | Aynı eğitim kayıtlarının kelime/offset ve tür başına BIO gösterimi |
| [`processed/crf_validation.jsonl`](processed/crf_validation.jsonl) | Aynı validation kayıtlarının kelime/offset ve tür başına BIO gösterimi |
| [`processed/crf_test.jsonl`](processed/crf_test.jsonl) | Aynı resmî test kayıtlarının kelime/offset ve tür başına BIO gösterimi |
| [`splits/split_manifest_v1.csv`](splits/split_manifest_v1.csv) | Kaynak/içerik gruplarını aynı bölümde tutan sabit üyelik listesi |
| [`splits/split_summary_v1.json`](splits/split_summary_v1.json) | Seed, bölüm sayıları ve özgün train/validation dosya hash'leri |
| [`samples/sample_records.jsonl`](samples/sample_records.jsonl) | Küçük, okunabilir train/validation örnekleri |
| [`samples/sample_crf.jsonl`](samples/sample_crf.jsonl) | Bu örneklerin kelime/BIO karşılığı |

İşlenmiş kayıtlar kalite işaretli bütün cevapları içerir. Model okuyucusunun
`quality == "good"` filtresi, ham/işlenmiş dosyalardan satır silinmesiyle aynı
şey değildir. Birebir tekrar anotasyonları ve örtüşmeler ham hedefte korunur;
aynı içerik gruplaması split sızıntısını önler, kaynakları silmez.

## Doğrulama ve yeniden üretim

Depo kökünden, veri indirmeden veya model çalıştırmadan teslim edilen
byte'ları doğrulamak için:

```bash
python scripts/verify_submission_data.py
python scripts/download_data.py --check-only
```

Boş bir veri dizininde yeniden üretim için:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

İlk komut sabit upstream revision'ı indirir, SHA-256 doğrular ve mevcut yanlış
dosyanın üzerine yazmaz. İkinci komut kayıtları birleştirir, orijinal cevap ve
anotasyon offset'lerini kontrol eder, manifesti doğrular ve model gösterimlerini
üretir. Resmî test dosyalarını üretmek için `prepare_data.py --include-test`
kullanılmıştır; bu adım tarihsel V1 kararları dondurulduktan sonra yapılmıştır.
Test etiketleri yeni model/eşik seçimi için kullanılmaz.

Bu teslim düzeltmesinde hazır yerel dosyalar yeniden işlenmeden eklenir;
orijinal split'ler, metinler, etiketler ve donmuş sonuçlar korunur. Train ve
validation hash'leri [nihai protokolde](../docs/FINAL_PROTOCOL.json) ve split
summary'de, resmî testin kimliği ise donmuş değerlendirme protokolünde
[`configs/final/final_evaluate.protocol.json`](../configs/final/final_evaluate.protocol.json)
bağlıdır. Yeni teslim doğrulayıcısı bu kimlikleri byte düzeyinde denetler;
test performansı hesaplamaz veya anotasyonlara göre ayar seçmez.

## Lisans ve ön işleme sınırı

Veri, [upstream MIT bildirimi](raw/LICENSE) ve kaynak atfıyla paylaşılır;
[docs/DATA_LICENSE.txt](../docs/DATA_LICENSE.txt) aynı bildirimin kopyasıdır.
Projenin kendi kaynak kodunun lisansı ayrı [LICENSE](../LICENSE) dosyasındadır.

Yaptığımız işlemler kaynak/cevap birleştirme, kalite filtresi, metadata/offset
doğrulama, tekrar/örtüşme analizi, kaynak gruplu bölümleme ve model hizalamasıdır.
Yeni anotasyon üretildiği, ham metinlerin kapsamlı biçimde elle temizlendiği veya
upstream veri setindeki etiket hatalarının giderildiği iddia edilmez. Detaylar:
[DATA_FORMAT.md](../docs/DATA_FORMAT.md) · [DATA_REPORT.md](../docs/DATA_REPORT.md).
