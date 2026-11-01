# RAGTruth project comparison — test

Metric: `ragtruth-project-eval-v1`. Quality filter: `good`. Original response character offsets; whitespace included. Project metrics; not a claim of paper-scorer equivalence.

| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |
|---|---:|---:|---:|---:|---:|---:|
| bce | 0.4444 | 0.5751 | 0.5014 | 0.2723 | 0.0768 | 1271 |
| focal | 0.3222 | 0.6243 | 0.4250 | 0.2930 | 0.0248 | 1271 |
| structured | 0.0490 | 1.0000 | 0.0935 | 0.1201 | 0.0692 | 1271 |

## Type support and performance

| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |
|---|---|---:|---:|---:|---:|---:|
| bce | Evident Conflict | 0.5119 | 13167 | 262 | 10108 | 0.1605 |
| bce | Subtle Conflict | 0.0038 | 1141 | 16 | 5129 | 0.0000 |
| bce | Evident Baseless Info | 0.3949 | 25011 | 323 | 39772 | 0.0841 |
| bce | Subtle Baseless Info | 0.1784 | 11270 | 153 | 23919 | 0.0405 |
| focal | Evident Conflict | 0.4760 | 13167 | 262 | 12440 | 0.0951 |
| focal | Subtle Conflict | 0.0040 | 1141 | 16 | 65346 | 0.0000 |
| focal | Evident Baseless Info | 0.4213 | 25011 | 323 | 33064 | 0.0618 |
| focal | Subtle Baseless Info | 0.2706 | 11270 | 153 | 24957 | 0.0335 |
| structured | Evident Conflict | 0.2306 | 13167 | 262 | 2771 | 0.1466 |
| structured | Subtle Conflict | 0.0022 | 1141 | 16 | 1032051 | 0.0000 |
| structured | Evident Baseless Info | 0.1996 | 25011 | 323 | 6707 | 0.1298 |
| structured | Subtle Baseless Info | 0.0478 | 11270 | 153 | 615 | 0.1365 |

## Breakdown: task_type

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| bce | Data2txt | 450 | 0.5429 | 0.3413 |
| bce | QA | 419 | 0.5255 | 0.1298 |
| bce | Summary | 402 | 0.2488 | 0.0976 |
| focal | Data2txt | 450 | 0.4773 | 0.3202 |
| focal | QA | 419 | 0.4969 | 0.1883 |
| focal | Summary | 402 | 0.1901 | 0.0894 |
| structured | Data2txt | 450 | 0.0938 | 0.1902 |
| structured | QA | 419 | 0.1323 | 0.0466 |
| structured | Summary | 402 | 0.0493 | 0.0005 |

## Breakdown: generator

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| bce | gpt-3.5-turbo-0613 | 211 | 0.2956 | 0.2305 |
| bce | gpt-4-0613 | 211 | 0.3284 | 0.1588 |
| bce | llama-2-13b-chat | 213 | 0.5641 | 0.3198 |
| bce | llama-2-70b-chat | 212 | 0.4794 | 0.2650 |
| bce | llama-2-7b-chat | 211 | 0.4629 | 0.2313 |
| bce | mistral-7B-instruct | 213 | 0.5445 | 0.2961 |
| focal | gpt-3.5-turbo-0613 | 211 | 0.1515 | 0.1897 |
| focal | gpt-4-0613 | 211 | 0.1157 | 0.1212 |
| focal | llama-2-13b-chat | 213 | 0.4737 | 0.3333 |
| focal | llama-2-70b-chat | 212 | 0.4501 | 0.2993 |
| focal | llama-2-7b-chat | 211 | 0.4259 | 0.2888 |
| focal | mistral-7B-instruct | 213 | 0.4775 | 0.2901 |
| structured | gpt-3.5-turbo-0613 | 211 | 0.0136 | 0.0396 |
| structured | gpt-4-0613 | 211 | 0.0135 | 0.0788 |
| structured | llama-2-13b-chat | 213 | 0.1247 | 0.1856 |
| structured | llama-2-70b-chat | 212 | 0.1099 | 0.1733 |
| structured | llama-2-7b-chat | 211 | 0.1446 | 0.0753 |
| structured | mistral-7B-instruct | 213 | 0.1408 | 0.0778 |

Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, so a predicted union is not an exact match for two separate raw annotations.
