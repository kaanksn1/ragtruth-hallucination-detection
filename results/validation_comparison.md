# RAGTruth project comparison — validation

Metric: `ragtruth-project-eval-v1`. Quality filter: `good`. Original response character offsets; whitespace included. Project metrics; not a claim of paper-scorer equivalence.

| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |
|---|---:|---:|---:|---:|---:|---:|
| crf_source | 0.3652 | 0.4202 | 0.3908 | 0.1999 | 0.0641 | 2240 |
| crf_response_only | 0.2146 | 0.2501 | 0.2310 | 0.1313 | 0.0528 | 2240 |
| transformer_default | 0.7269 | 0.3221 | 0.4464 | 0.2244 | 0.1192 | 2240 |
| transformer | 0.5076 | 0.5780 | 0.5405 | 0.3201 | 0.0700 | 2240 |
| weighted_trial | 0.4407 | 0.6207 | 0.5154 | 0.2942 | 0.0465 | 2240 |

## Type support and performance

| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |
|---|---|---:|---:|---:|---:|---:|
| crf_source | Evident Conflict | 0.3090 | 25146 | 478 | 34357 | 0.0644 |
| crf_source | Subtle Conflict | 0.0000 | 830 | 24 | 55 | 0.0000 |
| crf_source | Evident Baseless Info | 0.3572 | 57762 | 573 | 74749 | 0.0535 |
| crf_source | Subtle Baseless Info | 0.1333 | 18808 | 264 | 21534 | 0.0910 |
| crf_response_only | Evident Conflict | 0.3071 | 25146 | 478 | 32896 | 0.0697 |
| crf_response_only | Subtle Conflict | 0.0000 | 830 | 24 | 28 | 0.0000 |
| crf_response_only | Evident Baseless Info | 0.1462 | 57762 | 573 | 75990 | 0.0200 |
| crf_response_only | Subtle Baseless Info | 0.0720 | 18808 | 264 | 15209 | 0.1056 |
| transformer_default | Evident Conflict | 0.3882 | 25146 | 478 | 8390 | 0.1572 |
| transformer_default | Subtle Conflict | 0.0000 | 830 | 24 | 0 | 0.0000 |
| transformer_default | Evident Baseless Info | 0.4801 | 57762 | 573 | 36439 | 0.1066 |
| transformer_default | Subtle Baseless Info | 0.0294 | 18808 | 264 | 552 | 0.0705 |
| transformer | Evident Conflict | 0.5153 | 25146 | 478 | 23813 | 0.0989 |
| transformer | Subtle Conflict | 0.0212 | 830 | 24 | 1533 | 0.0000 |
| transformer | Evident Baseless Info | 0.5234 | 57762 | 573 | 60036 | 0.1027 |
| transformer | Subtle Baseless Info | 0.2204 | 18808 | 264 | 59130 | 0.0357 |
| weighted_trial | Evident Conflict | 0.4763 | 25146 | 478 | 23867 | 0.0902 |
| weighted_trial | Subtle Conflict | 0.0108 | 830 | 24 | 8760 | 0.0000 |
| weighted_trial | Evident Baseless Info | 0.4990 | 57762 | 573 | 77626 | 0.0557 |
| weighted_trial | Subtle Baseless Info | 0.1908 | 18808 | 264 | 80952 | 0.0300 |

## Breakdown: task_type

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_source | Data2txt | 791 | 0.3805 | 0.2078 |
| crf_source | QA | 736 | 0.4764 | 0.1511 |
| crf_source | Summary | 713 | 0.1929 | 0.0603 |
| crf_response_only | Data2txt | 791 | 0.3384 | 0.1852 |
| crf_response_only | QA | 736 | 0.1871 | 0.0494 |
| crf_response_only | Summary | 713 | 0.0654 | 0.0235 |
| transformer_default | Data2txt | 791 | 0.4690 | 0.2754 |
| transformer_default | QA | 736 | 0.5096 | 0.1351 |
| transformer_default | Summary | 713 | 0.1584 | 0.0491 |
| transformer | Data2txt | 791 | 0.5558 | 0.3545 |
| transformer | QA | 736 | 0.5852 | 0.2115 |
| transformer | Summary | 713 | 0.3048 | 0.1278 |
| weighted_trial | Data2txt | 791 | 0.5238 | 0.3254 |
| weighted_trial | QA | 736 | 0.5617 | 0.1853 |
| weighted_trial | Summary | 713 | 0.3267 | 0.1238 |

## Breakdown: generator

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_source | gpt-3.5-turbo-0613 | 370 | 0.3287 | 0.1694 |
| crf_source | gpt-4-0613 | 372 | 0.0842 | 0.0443 |
| crf_source | llama-2-13b-chat | 377 | 0.4158 | 0.2400 |
| crf_source | llama-2-70b-chat | 377 | 0.3999 | 0.1822 |
| crf_source | llama-2-7b-chat | 375 | 0.4290 | 0.2060 |
| crf_source | mistral-7B-instruct | 369 | 0.3818 | 0.2026 |
| crf_response_only | gpt-3.5-turbo-0613 | 370 | 0.1623 | 0.0625 |
| crf_response_only | gpt-4-0613 | 372 | 0.0552 | 0.0252 |
| crf_response_only | llama-2-13b-chat | 377 | 0.2872 | 0.1766 |
| crf_response_only | llama-2-70b-chat | 377 | 0.2121 | 0.1285 |
| crf_response_only | llama-2-7b-chat | 375 | 0.2395 | 0.1309 |
| crf_response_only | mistral-7B-instruct | 369 | 0.2505 | 0.1474 |
| transformer_default | gpt-3.5-turbo-0613 | 370 | 0.5222 | 0.2527 |
| transformer_default | gpt-4-0613 | 372 | 0.2307 | 0.1223 |
| transformer_default | llama-2-13b-chat | 377 | 0.4734 | 0.2597 |
| transformer_default | llama-2-70b-chat | 377 | 0.4854 | 0.2388 |
| transformer_default | llama-2-7b-chat | 375 | 0.4294 | 0.2066 |
| transformer_default | mistral-7B-instruct | 369 | 0.4005 | 0.1986 |
| transformer | gpt-3.5-turbo-0613 | 370 | 0.5544 | 0.2917 |
| transformer | gpt-4-0613 | 372 | 0.2674 | 0.1868 |
| transformer | llama-2-13b-chat | 377 | 0.5551 | 0.3388 |
| transformer | llama-2-70b-chat | 377 | 0.5151 | 0.3076 |
| transformer | llama-2-7b-chat | 375 | 0.5492 | 0.3270 |
| transformer | mistral-7B-instruct | 369 | 0.5599 | 0.3274 |
| weighted_trial | gpt-3.5-turbo-0613 | 370 | 0.4646 | 0.2494 |
| weighted_trial | gpt-4-0613 | 372 | 0.2160 | 0.1437 |
| weighted_trial | llama-2-13b-chat | 377 | 0.5467 | 0.3269 |
| weighted_trial | llama-2-70b-chat | 377 | 0.4794 | 0.2804 |
| weighted_trial | llama-2-7b-chat | 375 | 0.5257 | 0.2866 |
| weighted_trial | mistral-7B-instruct | 369 | 0.5520 | 0.3125 |

Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, so a predicted union is not an exact match for two separate raw annotations.
