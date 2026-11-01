# CRF baseline — validation (seçim metriği)

Kapsam: tam train/validation koşumu.
Train: 12,702 cevap / 1,959,297 kelime token; validation: 2,240 cevap (yalnız quality = good).
Veri split_summary_v1 ile aynı: evet.
Metrik: karakter düzeyinde micro precision/recall/F1; yalnız ayar seçimi için. Projenin ortak scorer'ı 5. kişidedir. Resmî test kullanılmadı.

Viterbi en olası etiket dizisini seçer; `posterior_τ`, forward-backward marjinali P(B veya I) ≥ τ olan tokenları işaretler.

## Kaynak özellikli CRF — ayar ve çözümleme ızgarası

| Ayar | P (any) | R (any) | F1 (any) | F1 Evident Conflict | F1 Subtle Conflict | F1 Evident Baseless Info | F1 Subtle Baseless Info | Macro F1 (4 tür) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c1_0.1_c2_0.01, viterbi | 0.462 | 0.124 | 0.196 | 0.056 | 0.000 | 0.230 | 0.064 | 0.087 |
| c1_0.1_c2_0.01, posterior_0.5 | 0.426 | 0.175 | 0.248 | 0.164 | 0.000 | 0.253 | 0.088 | 0.126 |
| c1_0.1_c2_0.01, posterior_0.4 | 0.400 | 0.214 | 0.279 | 0.206 | 0.000 | 0.268 | 0.097 | 0.143 |
| c1_0.1_c2_0.01, posterior_0.3 | 0.369 | 0.268 | 0.310 | 0.236 | 0.000 | 0.290 | 0.113 | 0.160 |
| c1_0.1_c2_0.01, posterior_0.2 | 0.331 | 0.337 | 0.334 | 0.267 | 0.000 | 0.302 | 0.123 | 0.173 |
| c1_0.1_c2_0.01, posterior_0.15 | 0.300 | 0.381 | 0.336 | 0.270 | 0.000 | 0.297 | 0.130 | 0.174 |
| c1_0.1_c2_0.01, posterior_0.1 | 0.264 | 0.452 | 0.333 | 0.247 | 0.000 | 0.289 | 0.127 | 0.166 |
| c1_0.1_c2_0.01, posterior_0.05 | 0.214 | 0.557 | 0.309 | 0.203 | 0.000 | 0.260 | 0.128 | 0.148 |
| c1_0.1_c2_0.1, viterbi | 0.509 | 0.119 | 0.192 | 0.066 | 0.000 | 0.211 | 0.046 | 0.081 |
| c1_0.1_c2_0.1, posterior_0.5 | 0.461 | 0.172 | 0.251 | 0.174 | 0.000 | 0.254 | 0.079 | 0.127 |
| c1_0.1_c2_0.1, posterior_0.4 | 0.429 | 0.218 | 0.289 | 0.216 | 0.000 | 0.276 | 0.103 | 0.149 |
| c1_0.1_c2_0.1, posterior_0.3 | 0.391 | 0.271 | 0.320 | 0.249 | 0.000 | 0.299 | 0.129 | 0.169 |
| c1_0.1_c2_0.1, posterior_0.2 | 0.343 | 0.345 | 0.344 | 0.279 | 0.000 | 0.310 | 0.131 | 0.180 |
| c1_0.1_c2_0.1, posterior_0.15 | 0.311 | 0.401 | 0.350 | 0.281 | 0.000 | 0.307 | 0.129 | 0.179 |
| c1_0.1_c2_0.1, posterior_0.1 | 0.271 | 0.476 | 0.346 | 0.257 | 0.000 | 0.293 | 0.140 | 0.173 |
| c1_0.1_c2_0.1, posterior_0.05 | 0.213 | 0.585 | 0.312 | 0.203 | 0.000 | 0.264 | 0.135 | 0.150 |
| c1_0.1_c2_1, viterbi | 0.653 | 0.104 | 0.180 | 0.067 | 0.000 | 0.225 | 0.022 | 0.079 |
| c1_0.1_c2_1, posterior_0.5 | 0.596 | 0.171 | 0.266 | 0.181 | 0.000 | 0.291 | 0.053 | 0.131 |
| c1_0.1_c2_1, posterior_0.4 | 0.551 | 0.226 | 0.321 | 0.236 | 0.000 | 0.328 | 0.093 | 0.164 |
| c1_0.1_c2_1, posterior_0.3 | 0.481 | 0.281 | 0.355 | 0.269 | 0.000 | 0.349 | 0.109 | 0.182 |
| c1_0.1_c2_1, posterior_0.2 | 0.412 | 0.368 | 0.389 | 0.311 | 0.000 | 0.365 | 0.123 | 0.200 |
| c1_0.1_c2_1, posterior_0.15 | 0.365 | 0.420 | 0.391 | 0.309 | 0.000 | 0.357 | 0.133 | 0.200 |
| c1_0.1_c2_1, posterior_0.1 | 0.306 | 0.492 | 0.377 | 0.280 | 0.000 | 0.337 | 0.140 | 0.189 |
| c1_0.1_c2_1, posterior_0.05 | 0.231 | 0.627 | 0.338 | 0.217 | 0.000 | 0.294 | 0.157 | 0.167 |

Seçilen: **c1_0.1_c2_1, posterior_0.15** (en yüksek validation F1 any; eşitlikte tablodaki ilk satır).

## Tür başına bağımsız eşik (ek, `selected`i değiştirmez)

Tek paylaşılan eşik, nadir türü neredeyse hiç etkilemeyen birleşik 'any' F1 ile seçilir; bu arama her türün kendi F1'ini kullanır ve precision < 1% olan eşikleri (sıfır tahmin hariç) eler.

| Tür | Eşik | P | R | F1 | Tahmin edilen karakter | Gold karakter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Evident Conflict | 0.2 | 0.314 | 0.308 | 0.311 | 24,693 | 25,146 |
| Subtle Conflict | 0.5 | 0.000 | 0.000 | 0.000 | 0 | 830 |
| Evident Baseless Info | 0.2 | 0.359 | 0.372 | 0.365 | 59,750 | 57,762 |
| Subtle Baseless Info | 0.05 | 0.103 | 0.335 | 0.157 | 61,223 | 18,808 |

Birleşik (any, tür başına eşiklerle): F1 0.388, macro F1 0.208.

## Kaynak özelliklerinin katkısı (aynı ayar ve çözümleme, aynı veri)

| Ayar | P (any) | R (any) | F1 (any) | F1 Evident Conflict | F1 Subtle Conflict | F1 Evident Baseless Info | F1 Subtle Baseless Info | Macro F1 (4 tür) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kaynak + cevap | 0.365 | 0.420 | 0.391 | 0.309 | 0.000 | 0.357 | 0.133 | 0.200 |
| yalnız cevap | 0.215 | 0.250 | 0.231 | 0.307 | 0.000 | 0.146 | 0.072 | 0.131 |
| yalnız cevap, viterbi | 0.379 | 0.024 | 0.045 | 0.066 | 0.000 | 0.038 | 0.011 | 0.029 |

## Eğitim

| Ayar | Tür | İterasyon | Max'ta durdu | Süre (s) |
| --- | --- | ---: | :---: | ---: |
| c1_0.1_c2_0.01 | Evident Conflict | 150 | evet | 83.3 |
| c1_0.1_c2_0.01 | Subtle Conflict | 150 | evet | 88.8 |
| c1_0.1_c2_0.01 | Evident Baseless Info | 150 | evet | 83.8 |
| c1_0.1_c2_0.01 | Subtle Baseless Info | 150 | evet | 83.0 |
| c1_0.1_c2_0.1 | Evident Conflict | 150 | evet | 84.7 |
| c1_0.1_c2_0.1 | Subtle Conflict | 150 | evet | 85.2 |
| c1_0.1_c2_0.1 | Evident Baseless Info | 150 | evet | 83.4 |
| c1_0.1_c2_0.1 | Subtle Baseless Info | 150 | evet | 85.6 |
| c1_0.1_c2_1 | Evident Conflict | 150 | evet | 85.0 |
| c1_0.1_c2_1 | Subtle Conflict | 150 | evet | 84.4 |
| c1_0.1_c2_1 | Evident Baseless Info | 150 | evet | 82.8 |
| c1_0.1_c2_1 | Subtle Baseless Info | 150 | evet | 88.6 |

Toplam süre: 25.0 dk.
