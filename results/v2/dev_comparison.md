# RAGTruth project comparison — validation

Metric: `ragtruth-project-eval-v1`. Quality filter: `good`. Original response character offsets; whitespace included. Project metrics; not a claim of paper-scorer equivalence.

| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |
|---|---:|---:|---:|---:|---:|---:|
| bce | 0.4873 | 0.5539 | 0.5185 | 0.3056 | 0.0781 | 2240 |
| focal | 0.3568 | 0.6078 | 0.4496 | 0.2993 | 0.0261 | 2240 |
| structured | 0.0555 | 1.0000 | 0.1053 | 0.1237 | 0.0631 | 2240 |

## Type support and performance

| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |
|---|---|---:|---:|---:|---:|---:|
| bce | Evident Conflict | 0.5138 | 25146 | 478 | 19035 | 0.1545 |
| bce | Subtle Conflict | 0.0059 | 830 | 24 | 10443 | 0.0000 |
| bce | Evident Baseless Info | 0.4992 | 57762 | 573 | 66424 | 0.0947 |
| bce | Subtle Baseless Info | 0.2034 | 18808 | 264 | 47292 | 0.0394 |
| focal | Evident Conflict | 0.4894 | 25146 | 478 | 23885 | 0.0962 |
| focal | Subtle Conflict | 0.0022 | 830 | 24 | 115058 | 0.0002 |
| focal | Evident Baseless Info | 0.4828 | 57762 | 573 | 57820 | 0.0747 |
| focal | Subtle Baseless Info | 0.2230 | 18808 | 264 | 44237 | 0.0334 |
| structured | Evident Conflict | 0.2311 | 25146 | 478 | 4881 | 0.1633 |
| structured | Subtle Conflict | 0.0009 | 830 | 24 | 1843432 | 0.0000 |
| structured | Evident Baseless Info | 0.1999 | 57762 | 573 | 10047 | 0.0938 |
| structured | Subtle Baseless Info | 0.0628 | 18808 | 264 | 1202 | 0.1241 |

## Breakdown: task_type

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| bce | Data2txt | 791 | 0.5517 | 0.3535 |
| bce | QA | 736 | 0.5496 | 0.1905 |
| bce | Summary | 713 | 0.2946 | 0.1195 |
| focal | Data2txt | 791 | 0.4982 | 0.3318 |
| focal | QA | 736 | 0.5249 | 0.1931 |
| focal | Summary | 713 | 0.2285 | 0.0956 |
| structured | Data2txt | 791 | 0.0956 | 0.1969 |
| structured | QA | 736 | 0.1548 | 0.0498 |
| structured | Summary | 713 | 0.0667 | 0.0114 |

## Breakdown: generator

| Model | Group | Responses | Any character F1 | Typed macro F1 |
|---|---|---:|---:|---:|
| bce | gpt-3.5-turbo-0613 | 370 | 0.4853 | 0.2525 |
| bce | gpt-4-0613 | 372 | 0.2468 | 0.1171 |
| bce | llama-2-13b-chat | 377 | 0.5358 | 0.3366 |
| bce | llama-2-70b-chat | 377 | 0.4765 | 0.2820 |
| bce | llama-2-7b-chat | 375 | 0.5430 | 0.3111 |
| bce | mistral-7B-instruct | 369 | 0.5460 | 0.3127 |
| focal | gpt-3.5-turbo-0613 | 370 | 0.3064 | 0.2205 |
| focal | gpt-4-0613 | 372 | 0.1168 | 0.1199 |
| focal | llama-2-13b-chat | 377 | 0.4914 | 0.3380 |
| focal | llama-2-70b-chat | 377 | 0.4290 | 0.2787 |
| focal | llama-2-7b-chat | 375 | 0.4778 | 0.2951 |
| focal | mistral-7B-instruct | 369 | 0.5005 | 0.3158 |
| structured | gpt-3.5-turbo-0613 | 370 | 0.0308 | 0.0248 |
| structured | gpt-4-0613 | 372 | 0.0141 | 0.0640 |
| structured | llama-2-13b-chat | 377 | 0.1366 | 0.1707 |
| structured | llama-2-70b-chat | 377 | 0.1127 | 0.1504 |
| structured | llama-2-7b-chat | 375 | 0.1564 | 0.0989 |
| structured | mistral-7B-instruct | 369 | 0.1630 | 0.1028 |

Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, so a predicted union is not an exact match for two separate raw annotations.
