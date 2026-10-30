# Bağlam uzunluğu ve pencereleme — 4. kişiyle karar notu

Hazırlayan: 2. kişi. Model, `max_length` ve pencere kararı 4. kişiyle birlikte verilir. Kesinleştirme 1. kişinin GPU pilotundan sonra yapılır. Aşağıdakiler öneridir.

## Kelime uzunlukları (train + validation)

| Görev | Kaynak metni p95 / maks (kelime) |
|---|---:|
| QA | 510 / 640 |
| Data2txt | 1067 / 1785 |
| Summary | 1507 / 2080 |

Tamamı [DATA_REPORT.md](DATA_REPORT.md) dosyasında. Bunlar model token'ı değildir.

## Yaklaşık token ölçümü

Bu ortamda Hugging Face'e erişim olmadığı için ölçüm, train metinleri üzerinde eğitilmiş 30k'lık bir WordPiece tokenizer ile yapıldı (BERT benzeri). Kapsam: train + validation, `quality=good`, 14.942 cevap.

**Bu sayılar yaklaşıktır, karar için kullanılmamalıdır.** Gerçek tokenizer ile tekrar ölçülmeli; komut aşağıda.

| max_length | Birden çok pencere gereken cevap | Özellik / cevap | Tek pencerede kesilen kaynak (medyan / p95) | Cevabı bölünen | Gidiş-dönüşte kaybolan gold |
|---:|---:|---:|---:|---:|---:|
| 512 | %76,4 | ~5,6 | %38,8 / %78,9 | 44 | 0 |
| 1024 | %21,7 | ~1,25 | %0 / %36,3 | 0 | 0 |
| 2048 | %0,9 | ~1,01 | %0 / %0 | 0 | 0 |

**Yorum:**
- 512 token ile çoğu cevap 4–6 pencereye bölünür. Eğitim kabaca 5 kat uzar.
- Kanıtın olmadığı pencerede de gold etiket görüldüğü için hedef gürültülenir.
- Tek pencereyle kesmek, kaynağın ortalama üçte birini atar. Bu durum `PROTOCOL_DRAFT.md`'deki "eksik bağlam kanıtın yokluğunu taklit edebilir" uyarısına girer.

## Öneri

1. **Uzun bağlamlı küçük encoder, `max_length` 2048.** Örnek: `answerdotai/ModernBERT-base`, 8192 token destekler.
   - Pencere neredeyse hiç gerekmez.
   - Aynı görevde benzer kurulum: LettuceDetect ([arXiv 2502.17125](https://arxiv.org/abs/2502.17125)). ModernBERT ile 4096 token ve ikili token hedefi kullanıyor. Bağlam token'ları -100. Makalenin modelinden farklı bir mimaridir ve sonuçları bizim için hedef değildir.
2. **RX 7800 XT'de bellek yetmezse 1024 + pencere.**
   - Eğitimde `window_policy="best"` kullanılır. Pencere yalnız cevap ve kaynak metnine göre seçilir, gold kullanılmaz.
   - Çıkarımda bütün pencereler çalıştırılır, ardından `aggregate_window_scores(..., "min")` uygulanır.
3. **512 ancak bir karşılaştırma gerekiyorsa.** Kesilen kısım raporlanmalıdır.

## 4. kişiyle sabitlenecekler

| Ayar | Önerim | Sahibi |
|---|---|---|
| Encoder / tokenizer revision | ModernBERT-base (fast tokenizer şart) | 4 |
| `max_length` | 2048; bellek yetmezse 1024 | 4 + 1 (pilot) |
| `context_stride` | 128 | 2 + 4 |
| Hedef | `labels_multi` (4 tür, örtüşme korunur) veya `labels_bio_by_type` | 4 |
| Eğitim pencereleri | `best` | 2 + 4 |
| Pencereleri birleştirme | `min`; eşik validation'da seçilir | 2 + 4 |

## Gerçek tokenizer ile ölçüm (~5–6 dk, GPU gerekmez)

```powershell
python scripts/window_report.py --tokenizer answerdotai/ModernBERT-base --max-length 512 1024 2048
```

Sonuç `docs/window_report_<tokenizer>.json` dosyasına yazılır. Bu dosya PR'a eklenmeli ve bu sayfadaki tablo gerçek sayılarla güncellenmelidir. Resmî test okunmaz.

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
