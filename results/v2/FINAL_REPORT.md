# V2: yeni geliştirme turunun sonuçları

**Bu, RAGTruth resmî testinin tekrar ölçümü değildir.** Yeni kontrol kümesi, önceki projenin train bölümünden kaynak gruplarıyla ayrılmıştır. Bütün V2 modeller bu gruplar çıkarıldıktan sonra ilk pretrained ağırlıklardan yeniden eğitildi. Eski resmî sonuç %52,06 olarak arşivde kalır; aşağıdaki farklı kümenin skorları doğrudan bu değerle kıyaslanmaz.

Yeni holdout: 1271 good cevap / 1278 toplam. Hariç tutulan kalite etiketleri: {'truncated': 2, 'incorrect_refusal': 5}.

## Dev — model ve decoder seçiminde kullanıldı

| Model | Any character F1 (%) | Four-type macro F1 (%) | Exact typed-span F1 (%) |
|---|---:|---:|---:|
| bce | 51.85 | 30.56 | 7.81 |
| focal | 44.96 | 29.93 | 2.61 |
| structured | 10.53 | 12.37 | 6.31 |

## Yeni internal holdout — seçimler dondurulduktan sonra tek ölçüm

| Model | Any character F1 (%) | Four-type macro F1 (%) | Exact typed-span F1 (%) |
|---|---:|---:|---:|
| bce | 50.14 | 27.23 | 7.68 |
| focal | 42.50 | 29.30 | 2.48 |
| structured | 9.35 | 12.01 | 6.92 |

Dev ile seçilen birincil model: **bce**. Bu modelin yeni holdout dört-tür macro F1 farkı, aynı yeni train/holdout ayrımındaki fresh BCE kontrolüne göre **+0.00 yüzde puan**. Holdout tablosu sonradan başka bir modeli seçmek veya tekrar ayar yapmak için kullanılmadı.

## Ne değişti?

- BCE: dört bağımsız sigmoid çıkışlı kontrol.
- Focal: aynı mimari ve veri; gamma=2, alpha=0.75. Kolay hedeflerin loss katkısını azaltma hipotezi sınandı.
- Structured: aynı fresh encoder, orijinal kelime aralıklarına mean pooling, dört bağımsız BIO CRF. Türler arası örtüşmeler korunur. Orijinal metin veya anotasyonlar değiştirilmez.

Tablolardaki tamamlanmış her family üç epoch eğitildi ve bütün checkpointler saklandı. Model/decoder seçimi yalnız dev dört-tür character macro F1, ardından any F1, ardından erken epoch kuralıyla yapıldı. Focal/BCE için aynı 15 eşik kullanıldı. Structured için sabit dokuz O-bias değeri kullanıldı. Daha iyi bir sonuç varsayılmadı; negatif denemeler de tabloda tutuldu.

İptal edilen family/pilotlar: {}.

## Sınırlar

Dört türün tek tek skorları [holdout tablosunda](holdout_comparison.md). Any karakter, tür macro ve tam span ölçümleri aynı şey değildir. Nadir türlerde sıfır F1 kalabilir; modelin her türü güvenilir çözdüğü iddia edilmez. Tek seed ve bu sınırlı deney bütçesi çoklu-seed güven aralığı sağlamaz. Farklı decoder aramaları ve model mimarileri birlikte değiştiğinde kazanımın tamamı tek bir parçaya atfedilemez.

## Dosyalar

[Dev karşılaştırması](dev_comparison.md), [holdout karşılaştırması](holdout_comparison.md), [dev hata örnekleri](error_analysis.md), [grafikler](figures/holdout/). Demo kaydedilmiş dev tahminlerini gösterir; canlı inference değildir. Kaynaklar ve önceden ilan edilmiş kurallar [V2_PLAN.md](../../docs/V2_PLAN.md) içindedir.

Deneyin bilimsel kararı `docs/V2_FINAL_PROTOCOL.json`, byte-bound job çiftleri `configs/v2/final/` altındadır. Orijinal resmî test burada açılmadı veya yeniden skorlanmadı.
