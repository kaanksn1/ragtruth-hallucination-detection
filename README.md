# RAGTruth Halüsinasyon Tespiti

**NLP Proje 2 · Grup Anonymous**

Bu proje, kaynak metne dayanarak üretilen LLM cevaplarında kaynakla çelişen veya
kaynakta desteklenmeyen bölümleri karakter düzeyinde bulur ve dört türe ayırır.
Amaç genel bir doğruluk kontrolü değil, cevabın kaynağa bağlılığını ölçmektir.

## Veri

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) ([ACL 2024](https://aclanthology.org/2024.acl-long.585/)),
2.965 kaynak ve 17.790 LLM cevabı içerir. Dört tür: Evident Conflict, Subtle Conflict,
Evident Baseless Info ve Subtle Baseless Info.

Train ve validation kaynak grubuna göre ayrıldı; aynı kaynak iki tarafta birden yer almaz.
Ayrıntılar: [veri raporu](docs/DATA_REPORT.md).

## Yöntem

| Model | Yaklaşım |
|---|---|
| CRF (yalnız cevap) | Her tür için ayrı linear-chain CRF, kaynak özelliği yok |
| CRF (kaynaklı) | Aynı model, kaynak özellikleriyle |
| Transformer | ModernBERT-base, dört bağımsız çıktı, en fazla 2.048 token |

Dört bağımsız çıktı, farklı türdeki örtüşen etiketleri korur. Model ve eşik seçimi
validation ile yapıldı; kararlar resmî testten önce [donduruldu](docs/FINAL_PROTOCOL.json).

## Sonuçlar

Resmî test, `quality=good` olan 2.675 cevap üzerinde değerlendirildi:

| Model | Any-character F1 | Typed macro F1 | Exact typed-span F1 | Response-level F1* |
|---|---:|---:|---:|---:|
| CRF (yalnız cevap) | 0,2074 | 0,1195 | 0,0615 | 0,5878 |
| CRF (kaynaklı) | 0,3544 | 0,1976 | 0,0667 | 0,6624 |
| Transformer | **0,5206** | **0,2984** | **0,0678** | **0,7285** |

\* Sonradan eklenen açıklayıcı metrik; model seçiminde kullanılmadı.

Skorlar, proje scorer'ı dışında [bağımsız olarak yeniden üretildi](results/independent_rescore.md).

Modeller sorunlu bölgeleri kısmen buluyor, ancak span sınırlarını tam tutturmakta
zorlanıyor. Subtle Conflict F1 üç modelde de 0'dır; bu tür testte yalnızca 15 cevapta
geçer. Sonuçlar tek seed'e dayanır.

Ayrıntılar: [teknik rapor](results/FINAL_REPORT.md) · [Türkçe bulgular](results/BULGULAR_TR.md) ·
[hata analizi](results/error_analysis.md) · [etiket kalitesi](results/label_quality.md)

## Son geliştirmeler

- **Seçim düzeltmeleri:** Model seçiminde detection F1 önceliklendirildi, ortak eşik
  araması ve %1 kesinlik tabanı eklendi. Validation detection F1 değerleri
  Transformer'da %55,07, CRF'te %39,22 oldu. Yukarıdaki donmuş test sonuçları değişmedi.
  Bkz. [proje devri](PROJECT_HANDOFF.md).
- **Bağımsız doğrulama:** Test skorları yeniden üretildi ve etiket kalitesi denetlendi.
- **V2 denemeleri:** Focal loss, BCE ve yapılandırılmış modeller ayrı bir iç holdout ile
  [`feature/ragtruth-v2`](https://github.com/kaanksn1/ragtruth-hallucination-detection/tree/feature/ragtruth-v2)
  branch'inde tutulur. Bu sonuçlar resmî testle doğrudan karşılaştırılamaz.

## Kurulum

Python 3.12 gerekir. Veri indirme ve testler için ek paket gerekmez:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python -m unittest discover -s tests -v
```

Model bağımlılıkları `requirements-crf.txt` ve `requirements-transformer.txt` içindedir.
Eğitim ve değerlendirme adımları: [çalıştırma rehberi](docs/RUNNING.md) ·
[teknik teslim rehberi](docs/TECHNICAL_HANDOFF.md).

## Proje yapısı

| Yol | İçerik |
|---|---|
| `ragtruth_data/` | Veri şeması, bölümleme ve hizalama |
| `ragtruth_eval/` | Ortak değerlendirme |
| `scripts/` | Veri, eğitim, kalibrasyon ve analiz komutları |
| `configs/` | Model, deney ve dondurulmuş koşum ayarları |
| `demo/` | Yerel sonuç görüntüleyici |
| `results/` | Raporlar, metrikler ve grafikler |
| `tests/` | Birim ve entegrasyon testleri |
| `contributions/` | Üye katkı dosyaları |

## Ekip

| Üye | Katkı |
|---|---|
| Serdar Kaan Kesen | Merkezî koşum, entegrasyon, değerlendirme ve test |
| Samed Ortakaya | Veri sözleşmesi, bölümleme ve hizalama |
| Kemal Arif Karabacıoğlu | CRF baseline |
| Adnan Demir | Transformer ve seçim düzeltmeleri |
| Ozan Serhan Denizhan | Bağımsız doğrulama, etiket denetimi ve sunum |

Ayrıntılar: [katkı dosyaları](contributions/README.md).

## Lisans

RAGTruth veri seti MIT lisanslıdır: [lisans metni](docs/DATA_LICENSE.txt).
