# RAGTruth project comparison — test

Metric: `ragtruth-project-eval-v1`. Quality filter: `good`. Original response character offsets; whitespace included. Project metrics; not a claim of paper-scorer equivalence.

| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |
|---|---:|---:|---:|---:|---:|---:|
| crf_source | 0.3260 | 0.3883 | 0.3544 | 0.1976 | 0.0667 | 2675 |
| crf_response_only | 0.1832 | 0.2390 | 0.2074 | 0.1195 | 0.0615 | 2675 |
| transformer | 0.4828 | 0.5649 | 0.5206 | 0.2984 | 0.0678 | 2675 |

## Type support and performance

| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |
|---|---|---:|---:|---:|---:|---:|
| crf_source | Evident Conflict | 0.2715 | 24095 | 459 | 35206 | 0.0533 |
| crf_source | Subtle Conflict | 0.0000 | 558 | 15 | 390 | 0.0000 |
| crf_source | Evident Baseless Info | 0.3318 | 51143 | 542 | 56649 | 0.0632 |
| crf_source | Subtle Baseless Info | 0.1871 | 9558 | 141 | 18550 | 0.1221 |
| crf_response_only | Evident Conflict | 0.2768 | 24095 | 459 | 33642 | 0.0661 |
| crf_response_only | Subtle Conflict | 0.0000 | 558 | 15 | 460 | 0.0000 |
| crf_response_only | Evident Baseless Info | 0.1462 | 51143 | 542 | 65353 | 0.0439 |
| crf_response_only | Subtle Baseless Info | 0.0548 | 9558 | 141 | 16571 | 0.1089 |
| transformer | Evident Conflict | 0.4746 | 24095 | 459 | 21644 | 0.0801 |
| transformer | Subtle Conflict | 0.0000 | 558 | 15 | 1528 | 0.0000 |
| transformer | Evident Baseless Info | 0.5450 | 51143 | 542 | 45667 | 0.1349 |
| transformer | Subtle Baseless Info | 0.1740 | 9558 | 141 | 53758 | 0.0157 |

## Breakdown: task_type

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_source | Data2txt | 900 | 0.3398 | 0.1998 |
| crf_source | QA | 875 | 0.4587 | 0.1625 |
| crf_source | Summary | 900 | 0.1692 | 0.0573 |
| crf_response_only | Data2txt | 900 | 0.3241 | 0.1801 |
| crf_response_only | QA | 875 | 0.1443 | 0.0364 |
| crf_response_only | Summary | 900 | 0.0578 | 0.0181 |
| transformer | Data2txt | 900 | 0.5269 | 0.3189 |
| transformer | QA | 875 | 0.5839 | 0.2153 |
| transformer | Summary | 900 | 0.3260 | 0.1165 |

## Breakdown: generator

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| crf_source | gpt-3.5-turbo-0613 | 441 | 0.1224 | 0.0376 |
| crf_source | gpt-4-0613 | 450 | 0.1011 | 0.0458 |
| crf_source | llama-2-13b-chat | 450 | 0.3815 | 0.2245 |
| crf_source | llama-2-70b-chat | 449 | 0.3956 | 0.2220 |
| crf_source | llama-2-7b-chat | 449 | 0.3696 | 0.1851 |
| crf_source | mistral-7B-instruct | 436 | 0.3886 | 0.1985 |
| crf_response_only | gpt-3.5-turbo-0613 | 441 | 0.0882 | 0.0372 |
| crf_response_only | gpt-4-0613 | 450 | 0.0637 | 0.0294 |
| crf_response_only | llama-2-13b-chat | 450 | 0.2685 | 0.1667 |
| crf_response_only | llama-2-70b-chat | 449 | 0.1661 | 0.0937 |
| crf_response_only | llama-2-7b-chat | 449 | 0.2277 | 0.1291 |
| crf_response_only | mistral-7B-instruct | 436 | 0.2500 | 0.1391 |
| transformer | gpt-3.5-turbo-0613 | 441 | 0.2910 | 0.2060 |
| transformer | gpt-4-0613 | 450 | 0.2448 | 0.1228 |
| transformer | llama-2-13b-chat | 450 | 0.5296 | 0.3238 |
| transformer | llama-2-70b-chat | 449 | 0.5020 | 0.2745 |
| transformer | llama-2-7b-chat | 449 | 0.5368 | 0.2953 |
| transformer | mistral-7B-instruct | 436 | 0.5736 | 0.3210 |

Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, so a predicted union is not an exact match for two separate raw annotations.
