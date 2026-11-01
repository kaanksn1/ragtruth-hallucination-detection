# Bağımsız yeniden skorlama ve sınır analizi

Kod: `scripts/independent_rescore.py`. Projenin scorer'ı kullanılmadı; gold doğrudan
`data/raw/response.jsonl` dosyasından okundu. Tahminler 31 Ekim V1 paketindendir.
Hiçbir model, eşik veya protokol değişmedi. Ham sayılar: `independent_rescore.json`.

## Resmî test (2.675 cevap)

| Model | Any P | Any R | Any F1 | Typed macro F1 | Exact typed F1 | Aynı türde örtüşme F1 | Cevap düzeyi F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| crf_response_only | 0,1832 | 0,2390 | 0,2074 | 0,1195 | 0,0615 | 0,2616 | 0,5878 |
| crf_source | 0,3260 | 0,3883 | 0,3544 | 0,1976 | 0,0667 | 0,3184 | 0,6624 |
| transformer | 0,4828 | 0,5649 | 0,5206 | 0,2984 | 0,0678 | 0,3806 | 0,7285 |

İlk beş metrik `results/test_comparison.md` ile dört basamakta aynıdır. Örtüşme ve cevap
düzeyi F1, bu analizde sonradan eklenen açıklayıcı metriklerdir; model seçiminde
kullanılmadı. Eşikler cevap düzeyi için ayarlanmadı. Transformer cevap düzeyi accuracy
%78,88; her zaman "temiz" diyen sabit tahminin accuracy değeri %64,75'tir.

## Validation sınır analizi (yalnız validation, 2.240 cevap)

| Transformer | Tahmin span | Gold span | Exact typed F1 | Her iki taraf noktalamasız | Örtüşme F1 | Any F1 |
|---|---:|---:|---:|---:|---:|---:|
| Eşik 0,5 | 1.386 | 1.851 | 0,1192 | 0,1662 | 0,5109 | 0,4464 |
| Kalibre (nihai) | 4.552 | 1.851 | 0,0700 | 0,1087 | 0,3900 | 0,5405 |

- Noktalama kırpıldığında exact F1 artıyor ama küçük kalıyor; asıl fark noktalama değil.
- Yalnız tahmini kırpmak exact F1'i düşürüyor (0,0700 → 0,0653); kısa boşlukları
  birleştirmek anlamlı değiştirmiyor (0,0706). Son işlem önerilmez.
- Düşük eşikler any F1'i artırırken parçalanmayı büyütüyor: tahmin span sayısı gold'un
  yaklaşık 2,5 katı. Exact F1 düşüşünün görünen nedeni bu takastır.
