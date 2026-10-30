"""Shared constants for the RAGTruth data contract (version 1)."""

FORMAT_VERSION = "ragtruth-data-v1"

# The four annotation types, with their raw RAGTruth names, in a fixed order.
LABEL_TYPES = (
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "Subtle Baseless Info",
)

TASK_TYPES = ("QA", "Summary", "Data2txt")

# Only these record fields may be fed to a model. Everything else (ids,
# task_type, generator, quality, annotation metadata) is for matching,
# grouping, reporting or scoring — see docs/PROTOCOL_DRAFT.md.
MODEL_INPUT_FIELDS = ("source_text", "response")

# Binary BIO used for "is this token inside any annotated span".
BIO = ("O", "B", "I")
