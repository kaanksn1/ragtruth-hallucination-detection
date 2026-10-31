"""Project metrics for RAGTruth; not a reproduction of the paper's scorer."""
from .metrics import METRIC_VERSION, character_scores, evaluate_records, exact_span_scores
from .io import load_gold, read_predictions

__all__ = ["METRIC_VERSION", "character_scores", "evaluate_records", "exact_span_scores",
           "load_gold", "read_predictions"]
