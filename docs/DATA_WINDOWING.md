# Bağlam uzunluğu ve pencereleme — 4. kişiyle karar notu

Hazırlayan: 2. kişi. Model, `max_length` ve pencere kararı 4. kişiyle birlikte verilir. Kesinleştirme 1. kişinin GPU pilotundan sonra yapılır. Aşağıdakiler öneridir.

## Kelime uzunlukları (train + validation)

| Görev | Kaynak metni p95 / maks (kelime) |
|---|---:|
| QA | 510 / 640 |
| Data2txt | 1067 / 1785 |
| Summary | 1507 / 2080 |

Tamamı [DATA_REPORT.md](DATA_REPORT.md) içinde. Bunlar model token'ı değildir.

## Gerçek tokenizer ölçümü — `answerdotai/ModernBERT-base`

Komut: `scripts/window_report.py`, `context_stride=128`. Kapsam: train + validation, `quality=good`, 14.942 cevap. Resmî test okunmadı. Ham çıktı: [window_report_answerdotai__ModernBERT-base.json](window_report_answerdotai__ModernBERT-base.json).

| max_length | Birden çok pencere gereken cevap | Özellik / cevap | Tek pencerede kesilen kaynak (medyan / p95) | Cevabı bölünen | Gidiş-dönüşte kaybolan gold |
|---:|---:|---:|---:|---:|---:|
| 512 | %80,5 | 3,45 | %42,4 / %79,6 | 115 | 0 |
| 1024 | %21,9 | 1,25 | %0 / %36,7 | 0 | 0 |
| 2048 | **%0,9** | **1,01** | **%0 / %0** | 0 | 0 |

**Yorum:**

- **512 token:** Cevapların beşte dördü birden çok pencereye bölünüyor. Tek pencereyle kesilirse kaynağın medyanda %42'si atılıyor. Bu, `PROTOCOL_DRAFT.md`'deki "eksik bağlam kanıtın yokluğunu taklit edebilir" riskinin ta kendisi. Pencereleme eğitimi ~3,5 kat uzatıyor ve gürültü ekliyor: kanıtın olmadığı pencerede de gold etiket görülüyor.
- **1024 token:** Cevapların %22'si hâlâ pencere gerektiriyor.
- **2048 token:** Neredeyse her cevap kaynağıyla birlikte tek parçada sığıyor. Kaynağın hiçbir kısmı kesilmiyor.
- **Hizalama:** Her ayarda gold anotasyonların tamamı etiket → pencere → karakter span dönüşümünden sağlam çıkıyor.

## Öneri

1. **ModernBERT-base + `max_length` 2048.**
   - Pencere neredeyse hiç gerekmez. Kalan %0,9 için aynı pencere kodu kullanılır.
   - Aynı görevde benzer bir kurulum: LettuceDetect ([arXiv 2502.17125](https://arxiv.org/abs/2502.17125)). ModernBERT, 4096 token, ikili token hedefi; bağlam token'ları -100. Makalenin modelinden farklı bir mimaridir ve sonuçları bizim için hedef değildir.
2. **RX 7800 XT'de bellek yetmezse 1024 + pencere.**
   - Eğitimde `window_policy="best"`: pencere yalnız cevap ve kaynak metnine göre seçilir, gold kullanılmaz.
   - Çıkarımda bütün pencereler çalıştırılır ve `aggregate_window_scores(..., "min")` ile birleştirilir.
3. **512 yalnız karşılaştırma amaçlı.** Kesilen kısım raporlanmalı.

## 4. kişiyle sabitlenecekler

| Ayar | Önerim | Sahibi |
|---|---|---|
| Encoder / tokenizer revision | ModernBERT-base; revision commit'i model PR'ında sabitlenir | 4 |
| `max_length` | 2048; bellek yetmezse 1024 | 4 + 1 (pilot) |
| `context_stride` | 128 (en fazla yarım pencere) | 2 + 4 |
| Hedef | `labels_multi` (4 tür, örtüşme korunur) veya `labels_bio_by_type` | 4 |
| Eğitim pencereleri | `best` | 2 + 4 |
| Pencereleri birleştirme | `min`; eşik validation'da seçilir | 2 + 4 |

## Ölçümü tekrarlamak (GPU gerekmez, ~3 dk)

```powershell
python scripts/window_report.py --tokenizer answerdotai/ModernBERT-base --max-length 512 1024 2048
```

Başka bir tokenizer için aynı komutta `--tokenizer` değiştirilir. Sonuç `docs/window_report_<tokenizer>.json` dosyasına yazılır.

## Kullanım

```python
from transformers import AutoTokenizer
from ragtruth_data.data import load_split
from ragtruth_data.features import build_features, pad_batch

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
train = build_features(load_split("train"), tok, max_length=2048, window_policy="best")
val = build_features(load_split("validation"), tok, max_length=2048)          # tüm pencereler
batch = pad_batch(train[:8], tok.pad_token_id, label_key="labels_bio_by_type", return_tensors="pt")
```
