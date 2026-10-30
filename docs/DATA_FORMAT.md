# Ortak veri sözleşmesi (v1) — 2. kişi

`docs/TEAM.md` içindeki "Hazırlanmış örnek" önerisinin uygulanmış hâli. Bütün modeller (CRF, bağlamlı Transformer) ve scorer bu dosyalardan okur.

## Çalıştırma

```powershell
python scripts/download_data.py      # 1. kişinin doğrulamalı indirmesi (bir kez)
python scripts/prepare_data.py       # manifest kontrolü, train/validation dosyaları, rapor
python -m unittest discover -s tests -v
```

- Yalnız Python standart kütüphanesi kullanılır (3.12 ile test edildi). Çalışma süresi ~10 sn.
- Ayarlar `configs/data.json` içindedir: seed 42, validation %15, görev türü × kaynak veri seti tabakası.
- `data/splits/split_manifest_v1.csv` repoya commit'lidir ve **tek doğruluk kaynağıdır**. Script bölümü yeniden hesaplar; sonuç manifestle birebir aynı değilse durur.
- İşlenmiş dosyaların SHA-256 değerleri `data/splits/split_summary_v1.json` içindedir. Herkesin aynı veriyle çalıştığını testler kontrol eder.
- **Resmî test etiketleri okunmaz.** `--include-test` bayrağı yalnız protokol dondurulduktan sonraki tek final değerlendirme içindir.

| Çıktı | Git | İçerik |
|---|---|---|
| `data/processed/{train,validation}.jsonl` | hayır | Kayıtlar (şema aşağıda) |
| `data/processed/crf_{train,validation}.jsonl` | hayır | CRF için kelime token'ları ve etiketleri |
| `data/splits/split_manifest_v1.csv` | evet | `source_id, group_id, task_type, source_dataset, official_split, split, n_responses` |
| `data/splits/split_summary_v1.json` | evet | Sayılar, seed, ham ve işlenmiş dosya hash'leri |
| `data/samples/*.jsonl` | evet | 8 örnek kayıt (train/validation) |
| `docs/data_report.json`, `docs/DATA_REPORT.md` | evet | Kalite, tekrar, dört tür, örtüşme, uzunluk ve açıklayıcı örnekler |

## Bölüm kararları

| | Kaynak | Cevap |
|---|---:|---:|
| train | 2138 | 12.828 |
| validation | 377 | 2262 |
| test (resmî, dokunulmadı) | 450 | 2700 |

- **Kaynak bazlı ayrım:** Bir kaynağın 6 cevabı hep aynı taraftadır.
- **Aynı içerik gruplaması:** Farklı `source_id`'li ama aynı içerikli kaynaklar tek grup sayılır. Normalize edilmiş kaynak metni, prompt, Yelp işletmesi (ad + adres + şehir) veya QA soru/pasajları aynıysa iki kaynak aynı gruptadır. Bulunan gruplar: 12084/12085 ve 14041/14218; ikisi de tek tarafta. Resmî test kaynağıyla aynı içerikte olan bir kaynak bulunsaydı o da teste giderdi; bulunmadı.
- **Etiket kullanılmaz:** Ayrım yalnız kaynak içeriğine ve resmî split üyeliğine bakar.
- **Kalite filtresi:** Resmî train'de `incorrect_refusal` 120 ve `truncated` 28 cevap var. Bunlar dosyada durur. `load_split()` varsayılan olarak resmî baseline gibi yalnız `quality == "good"` olanları döndürür; filtrelemeden önceki ve sonraki sayılar `DATA_REPORT.md` içinde. Hiçbir tekrar silinmedi.

## Kayıt şeması (`{split}.jsonl`, satır başına bir cevap)

| Alan | Model girdisi mi? | Açıklama |
|---|---|---|
| `id` | hayır | RAGTruth cevap id'si (string) |
| `source_id` | hayır | Yalnız eşleştirme ve gruplama |
| `split`, `official_split` | hayır | Manifestteki bölüm / RAGTruth'un bölümü |
| `task_type`, `source_dataset`, `generator`, `temperature` | hayır | Rapor kırılımları |
| `quality` | hayır | `good` / `incorrect_refusal` / `truncated` |
| **`source_text`** | **evet** | Üretici modele verilen prompt, değiştirilmeden: talimat + (QA'da) soru + kaynak içerik |
| **`response`** | **evet** | Orijinal cevap; bütün offset'ler bu metne göredir |
| `labels` | hedef | Anotasyonlar (aşağıda) |
| `n_labels`, `has_hallucination` | hedef | Anotasyon sayısı; cevap düzeyi etiket |
| `source_info` | hayır | Ham yapılandırılmış kaynak; yalnız referans için |

`schema.MODEL_INPUT_FIELDS = ("source_text", "response")`. Diğer alanlar modele verilmez (`PROTOCOL_DRAFT.md`).

**`labels` öğesi:**

| Alan | Açıklama |
|---|---|
| `label_id` | Cevap içinde 0..n-1; `(start, end, tür)` sırasına göre |
| `start`, `end` | `response` içinde `[start, end)`; 12.756 eğitim anotasyonunun hepsinde `response[start:end] == text` |
| `label_type` | Ham adıyla: `Evident Conflict`, `Subtle Conflict`, `Evident Baseless Info`, `Subtle Baseless Info` |
| `text`, `meta`, `implicit_true`, `due_to_null` | RAGTruth'tan aynen. Model girdisi değildir |
| `orig_index` | Ham dosyadaki sırası |
| `duplicate_of` | Birebir aynı `(start, end, tür)` daha önce varsa onun `label_id`'si |
| `overlaps_with` | Karakter paylaştığı diğer anotasyonların `label_id` listesi |

Hiçbir anotasyon silinmez veya birleştirilmez. `implicit_true` ve `due_to_null` olan anotasyonlar da hedefte durur. Bunlar için ayrı bir politika seçilirse protokolde yazılmalıdır.

## Hizalama ve örtüşme politikası

- **Token kuralı:** Bir anotasyonla en az bir karakter paylaşan token o anotasyona aittir. Resmî train'de kelime ortasında başlayan veya biten anotasyon sayısı 12.756'da 7'dir.
- **Tür bazında hedef:** Her tür için ayrı bir `O/B/I` dizisi üretilir (`bio_by_type`). Farklı türdeki örtüşmeler (train'de 23 çift) böylece hiç kaybolmaz.
- **Aynı türde örtüşme:** Aynı türden örtüşen anotasyonlar etiketlerde birleşir (union). Etiketten geri dönüş bu birleşimi verir.
- **"Herhangi bir anotasyon" hedefi:** İkili `bio_any` dizisinde bütün türler birleşir.
- **Puanlama:** Ham `labels` alanı her zaman korunur ve puanlama bundan yapılmalıdır.

### CRF girdisi (`crf_{split}.jsonl`)

```json
{"id": "2", "tokens": ["New", "research", "..."], "offsets": [[0, 3], [4, 12]],
 "bio_any": ["O", "O"], "bio_by_type": {"Evident Conflict": ["O", "O"], "...": []},
 "split": "train", "quality": "good"}
```

- Token'lar `\w+(?:['’]\w+)*|[^\w\s]` deseniyle ayrılır.
- Etiketten span'e dönüş: `alignment.tags_to_char_spans(offsets, tags, label_type)`. Çıktı, `TEAM.md`'deki tahmin biçimindeki `{start, end, label_type}` yapısındadır.

### Transformer girdisi

`features.build_features(records, tokenizer, max_length, ...)` herhangi bir HF *fast* tokenizer ile çalışır.

- **Giriş sırası:** cevap + kaynak penceresi; cevap önce gelir. Hangi token'ın neye ait olduğu tokenizer'ın `sequence_ids()` çıktısından okunur.
- **Cevap kesilmez:** Cevap tek başına sığmazsa örtüşen parçalara bölünür.
- **Kaynak pencereleri:** Kaynak, `context_stride` token kadar örtüşen pencerelere bölünür. Kaynağın her karakteri en az bir pencerede görülür (test edildi).
- **Pencere seçimi yalnız model girdilerine bakar:** `window_policy="best"`, cevabın içerik kelimeleriyle en çok örtüşen kaynak penceresini seçer. Gold anotasyon kullanılmaz; bu da test edildi.
- **Etiketler:** Yalnız cevap token'larında doludur, diğer konumlarda `-100` olur:
  - `labels_any`: 0/1
  - `labels_bio`: O/B/I
  - `labels_multi`: 4 tür × 0/1, örtüşme korunur
  - `labels_bio_by_type`: tür başına O/B/I
- **Ek alanlar:** `response_mask`, `response_offsets` (mutlak karakter konumu), `source_char_span`.
- **Pencereden cevaba eşleme:** `aggregate_window_scores(len(response), features, token_scores, "min")` karakter başına skor verir. Her tür için ayrı çağrılır.
  - `min` kuralı: Kanıtı herhangi bir pencerede bulunan token işaretli kalmaz.
  - Bölünmüş cevap parçaları ortalanır.
  - `char_scores_to_spans` eşikle `{start, end, label_type}` span'lerine çevirir. Son dönüştürme kararı 4. kişinindir.

Pencere ölçümleri ve öneri: [DATA_WINDOWING.md](DATA_WINDOWING.md).

## Lisans

RAGTruth MIT lisanslıdır (© 2023 Particle Media). Örnek kayıtlar bu lisans kapsamında paylaşılır.
