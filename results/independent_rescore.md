# Bağımsız yeniden skorlama ve sınır analizi

Kod: `scripts/independent_rescore.py`. Projenin scorer matematiği kullanılmadı;
gold sabit SHA ile doğrulanan `data/raw/response.jsonl` dosyasından okundu.
Tahminler 31 Ekim V1'e aittir. Hiçbir model, eşik veya protokol değişmedi.

**2 Kasım doğrulaması:** Kohort tahmin ID'lerinden daraltılmaz; tam manifest/raw
kohortu ve `quality=good` filtresi kontrol edilir. Eksik/tekrar/yabancı ID,
bozuk tür/Unicode sınırı veya değişmiş dosya reddedilir. Test modu açık izin
ve donmuş test-tahmin kimlikleri gerektirir. Yeni doğrulanan ham sayılar:
[validation](review_2026-11-02/independent_validation.json),
[resmî test](review_2026-11-02/independent_test.json).
Eski `independent_rescore.json` ilk PR'ın tarihsel çıktısı olarak korunur.

## Resmî test (2.675 cevap)

| Model | Any P | Any R | Any F1 | Typed macro F1 | Exact typed F1 | Aynı türde örtüşme F1 | Cevap düzeyi F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| crf_response_only | 0,1832 | 0,2390 | 0,2074 | 0,1195 | 0,0615 | 0,2616 | 0,5878 |
| crf_source | 0,3260 | 0,3883 | 0,3544 | 0,1976 | 0,0667 | 0,3184 | 0,6624 |
| transformer | 0,4828 | 0,5649 | 0,5206 | 0,2984 | 0,0678 | 0,3806 | 0,7285 |

İlk beş metrik `results/test_comparison.md` ile aynıdır; yeni doğrulama tam
hassasiyetli JSON referanslarıyla `1e-12` toleransta eşleşti. Örtüşme ve cevap
düzeyi F1, bu analizde sonradan eklenen açıklayıcı metriklerdir; model seçiminde
kullanılmadı. Eşikler cevap düzeyi için ayarlanmadı. Transformer cevap düzeyi accuracy
%78,88; her zaman "temiz" diyen sabit tahminin accuracy değeri %64,75'tir.

Örtüşme sütunu aynı türde en az bir karakter paylaşan span'leri sayan **many-to-many
kapsama tanısıdır**. Bire bir eşleştirmeli span-F1 değildir; tek uzun tahmin birçok
gold span'e değebilir. Cevap düzeyi F1 ise yalnız cevapta hata var/yok kararını
ölçer. İki tanısal metrik de önceden ilan edilmiş ana başarı metriğinin yerine geçmez.

## Validation sınır analizi (yalnız validation, 2.240 cevap)

| Transformer | Tahmin span | Gold span | Exact typed F1 | Her iki taraf noktalamasız | Örtüşme F1 | Any F1 |
|---|---:|---:|---:|---:|---:|---:|
| Eşik 0,5 | 1.386 | 1.851 | 0,1192 | 0,1669 | 0,5109 | 0,4464 |
| Kalibre (nihai) | 4.552 | 1.851 | 0,0700 | 0,1096 | 0,3900 | 0,5405 |

- Noktalama kırpıldığında exact F1 artıyor ama küçük kalıyor; asıl fark noktalama değil.
- Yalnız tahmini kırpmak exact F1'i düşürüyor (0,0700 → 0,0658); kısa boşlukları
  birleştirmek anlamlı değiştirmiyor (0,0706). Son işlem önerilmez.
- Düşük eşikler any F1'i artırırken parçalanmayı büyütüyor: tahmin span sayısı gold'un
  yaklaşık 2,5 katı. Exact F1 düşüşünün görünen nedeni bu takastır.

Kırpma sonrası boş kalan aralıklar artık çıkarılır; geçersiz sıfır uzunluklu
span'ler sahte exact eşleşme üretmez. Bu düzeltme yalnız dev tanısal
kırpma sütunlarını etkiler. Ham exact/karakter/test skorları değişmedi.
Bu gözlem, bütün sınır hatalarının nedeninin kanıtlandığı veya başka
post-processing yöntemlerinin elendiği anlamına gelmez.

## Tekrar üretim

Gerekli sabit ham dosyalar ve yerel tahminler mevcutken standart Python yeterlidir:

```text
python scripts/independent_rescore.py --root . --pack . --boundary-diagnostics --output outputs/new-validation-audit.json
python scripts/independent_rescore.py --root . --pack . --split test --allow-test-report-only --output outputs/new-test-audit.json
```

İkinci komut yalnız önceden dondurulmuş tahminleri doğrular. Testte boundary
denemeleri reddedilir. Mevcut raporun üzerine yazılmaz. Taşınmış prediction
dosyası için `--predictions MODEL=PACK_RELATIVE_PATH` kullanılabilir; frozen
referansla aynı SHA zorunludur. Girdilerin SHA'ları ve script kimliği JSON'dadır.
