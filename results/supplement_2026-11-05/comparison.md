# RAGTruth project comparison — validation

Metric: `ragtruth-project-eval-v1`. Quality filter: `good`. Original response character offsets; whitespace included. Project metrics; not a claim of paper-scorer equivalence.

| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |
|---|---:|---:|---:|---:|---:|---:|
| crf_response_only | 0.2146 | 0.2501 | 0.2310 | 0.1313 | 0.0528 | 2240 |
| crf_source | 0.3652 | 0.4202 | 0.3908 | 0.1999 | 0.0641 | 2240 |
| modernbert_v1 | 0.5076 | 0.5780 | 0.5405 | 0.3201 | 0.0700 | 2240 |
| hmm | 0.1481 | 0.3616 | 0.2102 | 0.1094 | 0.0052 | 2240 |
| rnn | 0.1626 | 0.2035 | 0.1808 | 0.1058 | 0.0127 | 2240 |
| cnn | 0.1875 | 0.1924 | 0.1899 | 0.1148 | 0.0142 | 2240 |
| bert | 0.3738 | 0.4581 | 0.4116 | 0.2262 | 0.0203 | 2240 |

## Type support and performance

| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |
|---|---|---:|---:|---:|---:|---:|
| crf_response_only | Evident Conflict | 0.3071 | 25146 | 478 | 32896 | 0.0697 |
| crf_response_only | Subtle Conflict | 0.0000 | 830 | 24 | 28 | 0.0000 |
| crf_response_only | Evident Baseless Info | 0.1462 | 57762 | 573 | 75990 | 0.0200 |
| crf_response_only | Subtle Baseless Info | 0.0720 | 18808 | 264 | 15209 | 0.1056 |
| crf_source | Evident Conflict | 0.3090 | 25146 | 478 | 34357 | 0.0644 |
| crf_source | Subtle Conflict | 0.0000 | 830 | 24 | 55 | 0.0000 |
| crf_source | Evident Baseless Info | 0.3572 | 57762 | 573 | 74749 | 0.0535 |
| crf_source | Subtle Baseless Info | 0.1333 | 18808 | 264 | 21534 | 0.0910 |
| modernbert_v1 | Evident Conflict | 0.5153 | 25146 | 478 | 23813 | 0.0989 |
| modernbert_v1 | Subtle Conflict | 0.0212 | 830 | 24 | 1533 | 0.0000 |
| modernbert_v1 | Evident Baseless Info | 0.5234 | 57762 | 573 | 60036 | 0.1027 |
| modernbert_v1 | Subtle Baseless Info | 0.2204 | 18808 | 264 | 59130 | 0.0357 |
| hmm | Evident Conflict | 0.2179 | 25146 | 478 | 68159 | 0.0096 |
| hmm | Subtle Conflict | 0.0000 | 830 | 24 | 789 | 0.0000 |
| hmm | Evident Baseless Info | 0.1447 | 57762 | 573 | 163690 | 0.0000 |
| hmm | Subtle Baseless Info | 0.0751 | 18808 | 264 | 44861 | 0.0075 |
| rnn | Evident Conflict | 0.2898 | 25146 | 478 | 25956 | 0.0364 |
| rnn | Subtle Conflict | 0.0000 | 830 | 24 | 276 | 0.0000 |
| rnn | Evident Baseless Info | 0.1030 | 57762 | 573 | 105288 | 0.0039 |
| rnn | Subtle Baseless Info | 0.0304 | 18808 | 264 | 10726 | 0.0272 |
| cnn | Evident Conflict | 0.2976 | 25146 | 478 | 36135 | 0.0288 |
| cnn | Subtle Conflict | 0.0000 | 830 | 24 | 5 | 0.0000 |
| cnn | Evident Baseless Info | 0.0976 | 57762 | 573 | 69330 | 0.0060 |
| cnn | Subtle Baseless Info | 0.0641 | 18808 | 264 | 4133 | 0.0435 |
| bert | Evident Conflict | 0.3478 | 25146 | 478 | 23974 | 0.0454 |
| bert | Subtle Conflict | 0.0024 | 830 | 24 | 11487 | 0.0000 |
| bert | Evident Baseless Info | 0.3888 | 57762 | 573 | 40858 | 0.0138 |
| bert | Subtle Baseless Info | 0.1656 | 18808 | 264 | 99202 | 0.0155 |

## Breakdown: task_type

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_response_only | Data2txt | 791 | 0.3384 | 0.1852 |
| crf_response_only | QA | 736 | 0.1871 | 0.0494 |
| crf_response_only | Summary | 713 | 0.0654 | 0.0235 |
| crf_source | Data2txt | 791 | 0.3805 | 0.2078 |
| crf_source | QA | 736 | 0.4764 | 0.1511 |
| crf_source | Summary | 713 | 0.1929 | 0.0603 |
| modernbert_v1 | Data2txt | 791 | 0.5558 | 0.3545 |
| modernbert_v1 | QA | 736 | 0.5852 | 0.2115 |
| modernbert_v1 | Summary | 713 | 0.3048 | 0.1278 |
| hmm | Data2txt | 791 | 0.2462 | 0.0816 |
| hmm | QA | 736 | 0.2189 | 0.0653 |
| hmm | Summary | 713 | 0.0269 | 0.0110 |
| rnn | Data2txt | 791 | 0.2872 | 0.1369 |
| rnn | QA | 736 | 0.1529 | 0.0377 |
| rnn | Summary | 713 | 0.0493 | 0.0121 |
| cnn | Data2txt | 791 | 0.3206 | 0.1707 |
| cnn | QA | 736 | 0.1077 | 0.0255 |
| cnn | Summary | 713 | 0.0300 | 0.0064 |
| bert | Data2txt | 791 | 0.3828 | 0.1915 |
| bert | QA | 736 | 0.4820 | 0.1705 |
| bert | Summary | 713 | 0.0614 | 0.0451 |

## Breakdown: generator

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_response_only | gpt-3.5-turbo-0613 | 370 | 0.1623 | 0.0625 |
| crf_response_only | gpt-4-0613 | 372 | 0.0552 | 0.0252 |
| crf_response_only | llama-2-13b-chat | 377 | 0.2872 | 0.1766 |
| crf_response_only | llama-2-70b-chat | 377 | 0.2121 | 0.1285 |
| crf_response_only | llama-2-7b-chat | 375 | 0.2395 | 0.1309 |
| crf_response_only | mistral-7B-instruct | 369 | 0.2505 | 0.1474 |
| crf_source | gpt-3.5-turbo-0613 | 370 | 0.3287 | 0.1694 |
| crf_source | gpt-4-0613 | 372 | 0.0842 | 0.0443 |
| crf_source | llama-2-13b-chat | 377 | 0.4158 | 0.2400 |
| crf_source | llama-2-70b-chat | 377 | 0.3999 | 0.1822 |
| crf_source | llama-2-7b-chat | 375 | 0.4290 | 0.2060 |
| crf_source | mistral-7B-instruct | 369 | 0.3818 | 0.2026 |
| modernbert_v1 | gpt-3.5-turbo-0613 | 370 | 0.5544 | 0.2917 |
| modernbert_v1 | gpt-4-0613 | 372 | 0.2674 | 0.1868 |
| modernbert_v1 | llama-2-13b-chat | 377 | 0.5551 | 0.3388 |
| modernbert_v1 | llama-2-70b-chat | 377 | 0.5151 | 0.3076 |
| modernbert_v1 | llama-2-7b-chat | 375 | 0.5492 | 0.3270 |
| modernbert_v1 | mistral-7B-instruct | 369 | 0.5599 | 0.3274 |
| hmm | gpt-3.5-turbo-0613 | 370 | 0.1008 | 0.0416 |
| hmm | gpt-4-0613 | 372 | 0.0342 | 0.0112 |
| hmm | llama-2-13b-chat | 377 | 0.2532 | 0.1456 |
| hmm | llama-2-70b-chat | 377 | 0.1870 | 0.0906 |
| hmm | llama-2-7b-chat | 375 | 0.2663 | 0.1651 |
| hmm | mistral-7B-instruct | 369 | 0.2810 | 0.1476 |
| rnn | gpt-3.5-turbo-0613 | 370 | 0.0776 | 0.0393 |
| rnn | gpt-4-0613 | 372 | 0.0296 | 0.0111 |
| rnn | llama-2-13b-chat | 377 | 0.2204 | 0.1326 |
| rnn | llama-2-70b-chat | 377 | 0.1676 | 0.1015 |
| rnn | llama-2-7b-chat | 375 | 0.2063 | 0.1179 |
| rnn | mistral-7B-instruct | 369 | 0.2262 | 0.1255 |
| cnn | gpt-3.5-turbo-0613 | 370 | 0.0763 | 0.0381 |
| cnn | gpt-4-0613 | 372 | 0.0478 | 0.0197 |
| cnn | llama-2-13b-chat | 377 | 0.2510 | 0.1676 |
| cnn | llama-2-70b-chat | 377 | 0.1845 | 0.1164 |
| cnn | llama-2-7b-chat | 375 | 0.1859 | 0.1211 |
| cnn | mistral-7B-instruct | 369 | 0.2480 | 0.1333 |
| bert | gpt-3.5-turbo-0613 | 370 | 0.4305 | 0.1879 |
| bert | gpt-4-0613 | 372 | 0.0801 | 0.0299 |
| bert | llama-2-13b-chat | 377 | 0.4396 | 0.2447 |
| bert | llama-2-70b-chat | 377 | 0.3804 | 0.2126 |
| bert | llama-2-7b-chat | 375 | 0.4546 | 0.2369 |
| bert | mistral-7B-instruct | 369 | 0.3827 | 0.2173 |

Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, so a predicted union is not an exact match for two separate raw annotations.
