# RAGTruth veri raporu (train + validation)

> `scripts/prepare_data.py` tarafından otomatik üretilir; elle düzenlemeyin.
> Ham veri: RAGTruth `c103204b9ce2` · split seed 42 · validation oranı 0.15 · format `ragtruth-data-v1`
> Resmî testin anotasyonları incelenmedi; yalnız kaynak/cevap sayıları ve split üyeliği kullanıldı.

## Bölümler

| Bölüm | Kaynak | Cevap | quality=good | Anotasyonlu cevap (good) | Anotasyon | Anotasyonlu kelime oranı |
|---|---:|---:|---:|---:|---:|---:|
| train | 2138 | 12828 | 12702 | 5696 (%44.8) | 10837 | %5.6 |
| validation | 377 | 2262 | 2240 | 1002 (%44.7) | 1856 | %5.7 |
| test (resmî) | 450 | 2700 | — | — | — | — |

Kalite işaretleri: train: good 12702, incorrect_refusal 101, truncated 25; validation: good 2240, incorrect_refusal 19, truncated 3.
`good` dışındaki cevaplar dosyada kalır; `load_split()` varsayılan olarak onları resmî baseline gibi dışarıda bırakır.

## Dört tür (quality=good)

| Tür | train anotasyon | validation anotasyon | train kelime | validation kelime |
|---|---:|---:|---:|---:|
| Evident Conflict | 4017 | 670 | 29301 | 5196 |
| Subtle Conflict | 160 | 25 | 1364 | 155 |
| Evident Baseless Info | 4654 | 813 | 59866 | 11137 |
| Subtle Baseless Info | 2006 | 348 | 18929 | 3585 |

## Görev türüne göre (quality=good)

| Bölüm | Görev | Cevap | Anotasyonlu cevap oranı | Anotasyon |
|---|---|---:|---:|---:|
| train | QA | 4156 | %31.9 | 2289 |
| train | Summary | 4042 | %31.1 | 1526 |
| train | Data2txt | 4504 | %69.1 | 7022 |
| validation | QA | 736 | %29.8 | 351 |
| validation | Summary | 713 | %31.4 | 299 |
| validation | Data2txt | 791 | %70.7 | 1206 |

## Örtüşme ve tekrarlar

| | train | validation |
|---|---:|---:|
| Örtüşen anotasyonlu cevap | 80 | 18 |
| Örtüşen çift — aynı tür | 69 | 12 |
| Örtüşen çift — farklı tür | 23 | 6 |
| Birebir tekrar eden anotasyon | 48 | 5 |
| implicit_true | 1557 | 257 |
| due_to_null | 1226 | 214 |

- Resmî train'de birebir aynı cevap metinlerinin fazladan kopyası: 211 (bunların 60 tanesi aynı kaynağın cevapları; çoğu "Unable to answer…" reddi).
- Farklı kaynak gruplarında geçen aynı cevap metni: 2. Silinmedi.
- Farklı source_id ama aynı içerik: 12084/12085; 14041/14218 → her grup tek tarafta tutuldu.

## Uzunluk (regex kelime; model token'ı değildir)

| Görev | Cevap medyan / p95 / maks | Kaynak metni medyan / p95 / maks |
|---|---:|---:|
| QA | 136 / 272 / 436 | 320 / 510 / 640 |
| Summary | 116 / 243 / 732 | 508 / 1507 / 2080 |
| Data2txt | 173 / 286 / 395 | 710 / 1067 / 1785 |

Tokenizer uzunluğu: not measured here; run scripts/window_report.py with the chosen tokenizer

## Üretici modele göre anotasyonlu cevap oranı (train)

| Üretici | Oran |
|---|---:|
| gpt-3.5-turbo-0613 | %14.0 |
| gpt-4-0613 | %14.9 |
| llama-2-13b-chat | %58.2 |
| llama-2-70b-chat | %48.6 |
| llama-2-7b-chat | %64.0 |
| mistral-7B-instruct | %68.9 |

## Açıklayıcı eğitim örnekleri

**same-type overlap** — id `2306` (Summary)

- [0] `622:812` Evident Baseless Info, örtüşür: [1] — “The study is the first to track adult drinking patterns at the county level in the United States. In 2012, 18 percent of Americans were considered binge drinkers, up from 14 percent in 2005.”
- [1] `785:812` Evident Baseless Info, örtüşür: [0] — “up from 14 percent in 2005.”

**cross-type overlap** — id `2183` (Summary)

- [0] `551:569` Evident Conflict, örtüşür: [1] — “The birth of a boy”
- [1] `551:687` Evident Baseless Info, örtüşür: [0] — “The birth of a boy would also be celebrated, but there is a sense of hope for a girl to balance out the gender dynamics in the monarchy.”

**single label** — id `5660` (Data2txt)

- [0] `129:146` Evident Baseless Info — “vegetarian dishes”
