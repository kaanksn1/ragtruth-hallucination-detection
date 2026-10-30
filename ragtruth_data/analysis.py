"""Data report for train and validation only.

The official test set is not inspected: only its source/response counts and
split membership are reported (docs/PROTOCOL_DRAFT.md).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .alignment import bio_by_type, bio_tags, word_tokenize
from .schema import LABEL_TYPES, TASK_TYPES
from .splits import normalise_text

REPORT_SPLITS = ("train", "validation")


def _q(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[int((len(v) - 1) * p)]


def _lengths(values):
    return {"median": _q(values, .5), "p95": _q(values, .95), "max": max(values) if values else None}


def split_stats(rows: list) -> dict:
    good = [r for r in rows if r["quality"] == "good"]
    labels = [l for r in good for l in r["labels"]]
    same = cross = 0
    with_overlap = 0
    for r in good:
        has = False
        for l in r["labels"]:
            for o in l["overlaps_with"]:
                if o > l["label_id"]:
                    has = True
                    same += r["labels"][o]["label_type"] == l["label_type"]
                    cross += r["labels"][o]["label_type"] != l["label_type"]
        with_overlap += has

    tokens = positive = 0
    per_type_pos = Counter()
    for r in good:
        _, offs = word_tokenize(r["response"])
        tokens += len(offs)
        positive += sum(t != "O" for t in bio_tags(offs, r["labels"]))
        for t, tags in bio_by_type(offs, r["labels"]).items():
            per_type_pos[t] += sum(x != "O" for x in tags)

    by_task = {}
    for t in TASK_TYPES:
        g = [r for r in good if r["task_type"] == t]
        if g:
            h = sum(r["has_hallucination"] for r in g)
            by_task[t] = {"responses": len(g), "with_labels": h, "rate": round(h / len(g), 4),
                          "labels": sum(r["n_labels"] for r in g)}
    by_gen = {}
    for m in sorted({r["generator"] for r in good}):
        g = [r for r in good if r["generator"] == m]
        by_gen[m] = round(sum(r["has_hallucination"] for r in g) / len(g), 4)

    return {
        "sources": len({r["source_id"] for r in rows}),
        "responses": len(rows),
        "quality_flags": dict(Counter(r["quality"] for r in rows)),
        "good_responses": len(good),
        "good_with_labels": sum(r["has_hallucination"] for r in good),
        "good_label_rate": round(sum(r["has_hallucination"] for r in good) / max(1, len(good)), 4),
        "labels": len(labels),
        "labels_by_type": {t: sum(l["label_type"] == t for l in labels) for t in LABEL_TYPES},
        "exact_duplicate_labels": sum(l["duplicate_of"] is not None for l in labels),
        "implicit_true_labels": sum(l["implicit_true"] for l in labels),
        "due_to_null_labels": sum(l["due_to_null"] for l in labels),
        "responses_with_overlap": with_overlap,
        "overlap_pairs_same_type": same,
        "overlap_pairs_cross_type": cross,
        "label_chars": _lengths([l["end"] - l["start"] for l in labels if l["duplicate_of"] is None]),
        "word_tokens": tokens,
        "word_tokens_in_any_label": positive,
        "word_token_positive_rate": round(positive / max(1, tokens), 4),
        "word_tokens_by_type": dict(per_type_pos),
        "by_task": by_task,
        "label_rate_by_generator": by_gen,
    }


def length_stats(rows: list) -> dict:
    out = {}
    for t in TASK_TYPES:
        g = [r for r in rows if r["task_type"] == t and r["quality"] == "good"]
        if not g:
            continue
        seen, src = set(), []
        for r in g:
            if r["source_id"] not in seen:
                seen.add(r["source_id"])
                src.append(len(word_tokenize(r["source_text"])[0]))
        out[t] = {"response_words": _lengths([len(word_tokenize(r["response"])[0]) for r in g]),
                  "source_words": _lengths(src)}
    return out


def duplicate_stats(rows: list, group_of: dict) -> dict:
    """Duplicates inside the official train set (train + validation)."""
    texts = Counter(r["response"] for r in rows)
    by_norm = defaultdict(set)
    for r in rows:
        by_norm[normalise_text(r["response"])].add(group_of[r["source_id"]])
    within = Counter((r["source_id"], r["response"]) for r in rows)
    groups = defaultdict(list)
    for sid, g in group_of.items():
        groups[g].append(sid)
    return {
        "exact_duplicate_response_excess": sum(c - 1 for c in texts.values() if c > 1),
        "of_which_inside_same_source": sum(c - 1 for c in within.values() if c > 1),
        "response_texts_shared_between_source_groups": sum(len(v) > 1 for v in by_norm.values()),
        "same_content_source_groups": sorted(sorted(v, key=int) for v in groups.values() if len(v) > 1),
    }


def explanatory_examples(rows: list, k: int = 3) -> list:
    """Deterministic short training examples: overlap, cross-type overlap, plain label."""
    good = sorted((r for r in rows if r["split"] == "train" and r["quality"] == "good" and len(r["response"]) < 900),
                  key=lambda r: int(r["id"]))
    picks, used = [], set()
    tests = [
        ("same-type overlap", lambda r: any(r["labels"][o]["label_type"] == l["label_type"]
                                            for l in r["labels"] for o in l["overlaps_with"])),
        ("cross-type overlap", lambda r: any(r["labels"][o]["label_type"] != l["label_type"]
                                             for l in r["labels"] for o in l["overlaps_with"])),
        ("single label", lambda r: r["n_labels"] == 1 and r["task_type"] == "Data2txt"),
    ]
    for name, pred in tests[:k]:
        for r in good:
            if r["id"] not in used and pred(r):
                used.add(r["id"])
                picks.append({"why": name, "id": r["id"], "task_type": r["task_type"],
                              "labels": [{k2: l[k2] for k2 in ("label_id", "start", "end", "label_type", "text", "overlaps_with")}
                                         for l in r["labels"]]})
                break
    return picks


def full_report(records: list, group_of: dict, test_counts: dict, tokenizer_note: str | None = None) -> dict:
    by = defaultdict(list)
    for r in records:
        by[r["split"]].append(r)
    return {
        "scope": "train and validation annotations only; official test: counts and split membership only",
        "splits": {s: split_stats(by[s]) for s in REPORT_SPLITS if by[s]},
        "official_test": test_counts,
        "lengths_words": length_stats([r for s in REPORT_SPLITS for r in by[s]]),
        "duplicates_official_train": duplicate_stats([r for s in REPORT_SPLITS for r in by[s]], group_of),
        "examples": explanatory_examples(records),
        "tokenizer_lengths": tokenizer_note or "not measured here; run scripts/window_report.py with the chosen tokenizer",
    }


def render_markdown(report: dict, summary: dict) -> str:
    S = report["splits"]
    sp = [s for s in REPORT_SPLITS if s in S]
    L = []
    a = L.append
    a("# RAGTruth veri raporu (train + validation)")
    a("")
    a("> `scripts/prepare_data.py` tarafından otomatik üretilir; elle düzenlemeyin.")
    a(f"> Ham veri: RAGTruth `{summary['revision'][:12]}` · split seed {summary['seed']} · "
      f"validation oranı {summary['validation_fraction']} · format `{summary['format_version']}`")
    a("> Resmî testin anotasyonları incelenmedi; yalnız kaynak/cevap sayıları ve split üyeliği kullanıldı.")
    a("")
    a("## Bölümler")
    a("")
    a("| Bölüm | Kaynak | Cevap | quality=good | Anotasyonlu cevap (good) | Anotasyon | Anotasyonlu kelime oranı |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for s in sp:
        x = S[s]
        a(f"| {s} | {x['sources']} | {x['responses']} | {x['good_responses']} | "
          f"{x['good_with_labels']} (%{100 * x['good_label_rate']:.1f}) | {x['labels']} | %{100 * x['word_token_positive_rate']:.1f} |")
    t = report["official_test"]
    a(f"| test (resmî) | {t['sources']} | {t['responses']} | — | — | — | — |")
    a("")
    a("Kalite işaretleri: " + "; ".join(
        f"{s}: " + ", ".join(f"{k} {v}" for k, v in sorted(S[s]["quality_flags"].items())) for s in sp) + ".")
    a("`good` dışındaki cevaplar dosyada kalır; `load_split()` varsayılan olarak onları resmî baseline gibi dışarıda bırakır.")
    a("")
    a("## Dört tür (quality=good)")
    a("")
    a("| Tür | " + " | ".join(f"{s} anotasyon" for s in sp) + " | " + " | ".join(f"{s} kelime" for s in sp) + " |")
    a("|---|" + "---:|" * (2 * len(sp)))
    for ty in LABEL_TYPES:
        a(f"| {ty} | " + " | ".join(str(S[s]["labels_by_type"][ty]) for s in sp) + " | "
          + " | ".join(str(S[s]["word_tokens_by_type"].get(ty, 0)) for s in sp) + " |")
    a("")
    a("## Görev türüne göre (quality=good)")
    a("")
    a("| Bölüm | Görev | Cevap | Anotasyonlu cevap oranı | Anotasyon |")
    a("|---|---|---:|---:|---:|")
    for s in sp:
        for task, v in S[s]["by_task"].items():
            a(f"| {s} | {task} | {v['responses']} | %{100 * v['rate']:.1f} | {v['labels']} |")
    a("")
    a("## Örtüşme ve tekrarlar")
    a("")
    a("| | " + " | ".join(sp) + " |")
    a("|---|" + "---:|" * len(sp))
    for key, label in [("responses_with_overlap", "Örtüşen anotasyonlu cevap"),
                       ("overlap_pairs_same_type", "Örtüşen çift — aynı tür"),
                       ("overlap_pairs_cross_type", "Örtüşen çift — farklı tür"),
                       ("exact_duplicate_labels", "Birebir tekrar eden anotasyon"),
                       ("implicit_true_labels", "implicit_true"),
                       ("due_to_null_labels", "due_to_null")]:
        a(f"| {label} | " + " | ".join(str(S[s][key]) for s in sp) + " |")
    d = report["duplicates_official_train"]
    a("")
    a(f"- Resmî train'de birebir aynı cevap metinlerinin fazladan kopyası: {d['exact_duplicate_response_excess']} "
      f"(bunların {d['of_which_inside_same_source']} tanesi aynı kaynağın cevapları; çoğu \"Unable to answer…\" reddi).")
    a(f"- Farklı kaynak gruplarında geçen aynı cevap metni: {d['response_texts_shared_between_source_groups']}. Silinmedi.")
    a("- Farklı source_id ama aynı içerik: " + ("; ".join("/".join(g) for g in d["same_content_source_groups"]) or "yok")
      + " → her grup tek tarafta tutuldu.")
    a("")
    a("## Uzunluk (regex kelime; model token'ı değildir)")
    a("")
    a("| Görev | Cevap medyan / p95 / maks | Kaynak metni medyan / p95 / maks |")
    a("|---|---:|---:|")
    for task, v in report["lengths_words"].items():
        r, c = v["response_words"], v["source_words"]
        a(f"| {task} | {r['median']} / {r['p95']} / {r['max']} | {c['median']} / {c['p95']} / {c['max']} |")
    a("")
    a(f"Tokenizer uzunluğu: {report['tokenizer_lengths']}")
    a("")
    a("## Üretici modele göre anotasyonlu cevap oranı (train)")
    a("")
    a("| Üretici | Oran |")
    a("|---|---:|")
    for m, v in S["train"]["label_rate_by_generator"].items():
        a(f"| {m} | %{100 * v:.1f} |")
    a("")
    a("## Açıklayıcı eğitim örnekleri")
    a("")
    for ex in report["examples"]:
        a(f"**{ex['why']}** — id `{ex['id']}` ({ex['task_type']})")
        a("")
        for l in ex["labels"]:
            ov = f", örtüşür: {l['overlaps_with']}" if l["overlaps_with"] else ""
            a(f"- [{l['label_id']}] `{l['start']}:{l['end']}` {l['label_type']}{ov} — “{l['text']}”")
        a("")
    return "\n".join(L)
