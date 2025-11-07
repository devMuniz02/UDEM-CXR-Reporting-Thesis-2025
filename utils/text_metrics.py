"""
Text metrics utilities for chest X-ray report evaluation.
Includes BLEU, ROUGE, METEOR, cosine similarity, BERTScore, CheXbert, and RadGraph metrics.

This version includes robust caching with test-friendly invalidation for CheXbert and RadGraph.
"""

# =========================
# Standard library imports
# =========================
from __future__ import annotations
import os
import pathlib
import re
import json

# =========================
# Typing imports
# =========================
from typing import Union, List, Tuple, Dict, Any, Sequence, Optional

# =========================
# Third-party imports
# =========================
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from torch import cuda

# Optional: BERTScore (domain-aware)
try:
    from bert_score import score as bertscore_score
except ImportError:
    bertscore_score = None

# RadGraph imports
try:
    from radgraph import F1RadGraph
except Exception as _e_rg:
    F1RadGraph = None
    _RG_IMPORT_ERR = _e_rg

# CheXbert imports (support several common module names)
_F1CHEXBERT_IMPORT_ERR = None
F1CheXbert = None
for _imp in ("f1chexbert", "chexbert", "radgraph.chexbert"):
    try:
        F1CheXbert = __import__(_imp, fromlist=["F1CheXbert"]).F1CheXbert
        break
    except Exception as _e_cx:
        _F1CHEXBERT_IMPORT_ERR = _e_cx

TextLike = Union[str, List[str], Sequence[str]]

# =========================
# Helpers
# =========================

def _normalize_text(s: str) -> str:
    """Collapse whitespace and strip ends to reduce spurious mismatches."""
    return re.sub(r"\s+", " ", s).strip()

def _as_lists(generated: TextLike, original: TextLike) -> Tuple[List[str], List[str]]:
    """
    Coerce inputs to aligned lists of the same length (min of the two).
    Returns normalized copies.
    """
    gen = [generated] if isinstance(generated, str) else list(generated)
    ref = [original] if isinstance(original, str) else list(original)
    gen = [_normalize_text(x) for x in gen]
    ref = [_normalize_text(x) for x in ref]
    n = min(len(gen), len(ref))
    if n == 0:
        return [], []
    return gen[:n], ref[:n]

def _has_cuda() -> bool:
    """Check if CUDA is available for torch."""
    try:
        return cuda.is_available()
    except Exception:
        return False

# =========================
# Classic text metrics
# =========================
def bleu_score(generated: TextLike, original: TextLike) -> List[float]:
    gen, ref = _as_lists(generated, original)
    smoothie = SmoothingFunction().method1
    return [float(sentence_bleu([r.split()], g.split(), smoothing_function=smoothie))
            for g, r in zip(gen, ref)]

def rouge_l_score(generated: TextLike, original: TextLike) -> List[float]:
    gen, ref = _as_lists(generated, original)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [float(scorer.score(r, g)['rougeL'].fmeasure) for g, r in zip(gen, ref)]

def cosine_sim_score(generated: TextLike, original: TextLike) -> List[float]:
    gen, ref = _as_lists(generated, original)
    if len(gen) == 0:
        return []
    vec = TfidfVectorizer().fit(gen + ref)
    out = []
    for g, r in zip(gen, ref):
        tfidf = vec.transform([g, r])
        out.append(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]))
    return out

def meteor_score_batch(generated: TextLike, original: TextLike) -> List[float]:
    gen, ref = _as_lists(generated, original)
    return [float(meteor_score([r.split()], g.split())) for g, r in zip(gen, ref)]

# =========================
# BERTScore (biomedical default)
# =========================
def bertscore_metric(
    generated: TextLike,
    original: TextLike,
    model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    device: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Returns dict with lists: {'p': [...], 'r': [...], 'f1': [...]}
    """
    if bertscore_score is None:
        return {"p": [], "r": [], "f1": []}
    gen, ref = _as_lists(generated, original)
    if len(gen) == 0:
        return {"p": [], "r": [], "f1": []}
    dev = device or ('cuda' if _has_cuda() else 'cpu')
    try:
        P, R, F1 = bertscore_score(gen, ref, model_type=model, device=dev, verbose=False)
    except KeyError:
        P, R, F1 = bertscore_score(gen, ref, model_type="bert-base-uncased", device=dev, verbose=False)
    return {
        "p":  P.cpu().numpy().tolist(),
        "r":  R.cpu().numpy().tolist(),
        "f1": F1.cpu().numpy().tolist()
    }

# =========================
# CheXbert (CheXpert-14)
# =========================

_CHEXPERT_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

# cache the scorer to avoid repeated weight loads
_CHEX_SCORER: Optional["F1CheXbert"] = None

def _default_chexbert_weights_path() -> str:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return str(pathlib.Path(local) / "chexbert" / "chexbert" / "Cache" / "chexbert.pth")
    return str(pathlib.Path.home() / ".cache" / "chexbert" / "chexbert.pth")

def chexbert_metrics(generated: TextLike, original: TextLike) -> Dict[str, Any]:
    """
    Compute CheXbert metrics for generated/reference report pairs.
    Wraps F1CheXbert to expose:
        - chexbert_f1_weighted: dataset-level weighted F1 (class_report['weighted avg'])
        - chexbert_f1_micro: dataset-level micro-F1
        - chexbert_f1_macro: dataset-level macro-F1
        - chexbert_f1_micro_5 / chexbert_f1_macro_5: top-5 subset if provided by the scorer
        - chexbert_per_pair_micro: list of per-pair micro-F1
        - chexbert_per_label_f1: length-14 vector (order = _CHEXPERT_14)
    """
    gen, ref = _as_lists(generated, original)

    # Empty input safeguard
    if len(gen) == 0:
        return {
            "chexbert_f1_weighted": None,
            "chexbert_f1_micro": None,
            "chexbert_f1_macro": None,
            "chexbert_f1_micro_5": None,
            "chexbert_f1_macro_5": None,
            "chexbert_per_pair_micro": [],
            "chexbert_per_label_f1": [],
            "chexbert_labels": _CHEXPERT_14,
        }

    # Ensure the scorer is available
    if F1CheXbert is None:
        print("[CheXbert] Import error:", repr(_F1CHEXBERT_IMPORT_ERR))
        return {
            "chexbert_f1_weighted": None,
            "chexbert_f1_micro": None,
            "chexbert_f1_macro": None,
            "chexbert_f1_micro_5": None,
            "chexbert_f1_macro_5": None,
            "chexbert_per_pair_micro": [],
            "chexbert_per_label_f1": [],
            "chexbert_labels": _CHEXPERT_14,
        }

    try:
        global _CHEX_SCORER
        # Invalidate cache if the class binding changed (important for tests that monkeypatch)
        if _CHEX_SCORER is not None and _CHEX_SCORER.__class__ is not F1CheXbert:
            _CHEX_SCORER = None

        if _CHEX_SCORER is None:
            _CHEX_SCORER = F1CheXbert()  # cache once

        # Expected outputs:
        #   accuracy, accuracy_not_averaged, class_report
        #   accuracy, accuracy_not_averaged, class_report, class_report_5
        out = _CHEX_SCORER(hyps=gen, refs=ref)
        if not isinstance(out, (list, tuple)) or len(out) < 3:
            raise RuntimeError("Unexpected F1CheXbert output structure.")

        if len(out) == 4:
            _, accuracy_not_averaged, class_report, class_report_5 = out
        else:
            _, accuracy_not_averaged, class_report = out
            class_report_5 = {}

        weighted_f1 = float(class_report.get("weighted avg", {}).get("f1-score", 0.0))
        micro_f1    = float(class_report.get("micro avg",    {}).get("f1-score", 0.0))
        macro_f1    = float(class_report.get("macro avg",    {}).get("f1-score", 0.0))

        micro_f1_5  = float(class_report_5.get("micro avg",  {}).get("f1-score", 0.0)) if class_report_5 else None
        macro_f1_5  = float(class_report_5.get("macro avg",  {}).get("f1-score", 0.0)) if class_report_5 else None

        per_label_f1 = [float(class_report.get(lbl, {}).get("f1-score", 0.0)) for lbl in _CHEXPERT_14]
        per_pair_micro = [float(x) for x in list(accuracy_not_averaged)]

        return {
            "chexbert_f1_weighted": weighted_f1,
            "chexbert_f1_micro": micro_f1,
            "chexbert_f1_macro": macro_f1,
            "chexbert_f1_micro_5": micro_f1_5,
            "chexbert_f1_macro_5": macro_f1_5,
            "chexbert_per_pair_micro": per_pair_micro,
            "chexbert_per_label_f1": per_label_f1,
            "chexbert_labels": _CHEXPERT_14,
        }

    except FileNotFoundError:
        # Reset cache so subsequent calls don’t reuse a bad/partial instance
        _CHEX_SCORER = None
        missing_path = _default_chexbert_weights_path()
        print("[CheXbert] Weights missing. Expected at:", missing_path)
        print("→ Place 'chexbert.pth' there and re-run.")
        return {
            "chexbert_f1_weighted": None,
            "chexbert_f1_micro": None,
            "chexbert_f1_macro": None,
            "chexbert_f1_micro_5": None,
            "chexbert_f1_macro_5": None,
            "chexbert_per_pair_micro": [],
            "chexbert_per_label_f1": [],
            "chexbert_labels": _CHEXPERT_14,
        }
    except Exception as e:
        # Catch-all to avoid crashing training loops
        print("[CheXbert] Unexpected error:", repr(e))
        return {
            "chexbert_f1_weighted": None,
            "chexbert_f1_micro": None,
            "chexbert_f1_macro": None,
            "chexbert_f1_micro_5": None,
            "chexbert_f1_macro_5": None,
            "chexbert_per_pair_micro": [],
            "chexbert_per_label_f1": [],
            "chexbert_labels": _CHEXPERT_14,
        }

# =========================
# RadGraph (entity/relation F1)
# =========================

_RG_SCORER: Optional["F1RadGraph"] = None

def radgraph_metric(generated: TextLike, original: TextLike) -> Tuple[float, float, float]:
    """
    Compute RadGraph entity/relation F1 metrics for generated/reference report pairs.
    Returns:
        (entity F1, entity+relation F1, bar entity+relation F1).
    """
    gen, ref = _as_lists(generated, original)
    if len(gen) == 0:
        return 0.0, 0.0, 0.0

    if F1RadGraph is None:
        raise ImportError(
            "Could not import F1RadGraph. Ensure the correct package is installed. "
            f"Original error: {repr(_RG_IMPORT_ERR)}"
        )

    global _RG_SCORER
    # Invalidate cache if the class binding changed (important for tests that monkeypatch)
    if _RG_SCORER is not None and _RG_SCORER.__class__ is not F1RadGraph:
        _RG_SCORER = None

    if _RG_SCORER is None:
        _RG_SCORER = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    # mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists
    mean_reward, *_ = _RG_SCORER(hyps=gen, refs=ref)
    rg_e, rg_er, rg_bar_er = mean_reward
    return float(rg_e), float(rg_er), float(rg_bar_er)

# =========================
# All metrics evaluation (kept for convenience; RadGraph/CheXbert used above)
# =========================

def evaluate_all_metrics(
    generated: TextLike,
    original: TextLike,
    evaluation_mode: str = "CheXagent"
) -> Dict[str, Any]:
    """
    Evaluate text metrics for generated/reference report pairs.
    If evaluation_mode == 'CheXagent', returns the subset commonly reported in those papers.
    """
    gen, ref = _as_lists(generated, original)

    cx = chexbert_metrics(gen, ref)
    rg_e, rg_er, _ = radgraph_metric(gen, ref)

    if evaluation_mode == "CheXagent":
        return {
            # CheXbert-focused (dataset-level)
            "chexbert_f1_weighted": cx["chexbert_f1_weighted"],
            "chexbert_f1_micro": cx["chexbert_f1_micro"],
            "chexbert_f1_macro": cx["chexbert_f1_macro"],
            "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],
            "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],
            # RadGraph F1
            "radgraph_f1_RG_E": rg_e,
            "radgraph_f1_RG_ER": rg_er,
        }

    bs = bertscore_metric(gen, ref)

    return {
        # classic
        "bleu": bleu_score(gen, ref),
        "rouge_l": rouge_l_score(gen, ref),
        "cosine_similarity": cosine_sim_score(gen, ref),
        "meteor": meteor_score_batch(gen, ref),
        # bertscore
        "bertscore_f1": bs["f1"],
        # RadGraph F1
        "radgraph_f1_RG_E": rg_e,
        "radgraph_f1_RG_ER": rg_er,
        # CheXbert
        "chexbert_f1_weighted": cx["chexbert_f1_weighted"],
        "chexbert_f1_micro": cx["chexbert_f1_micro"],
        "chexbert_f1_macro": cx["chexbert_f1_macro"],
        "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],
        "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],
        "chexbert_per_pair_micro": cx["chexbert_per_pair_micro"],
        "chexbert_per_label_f1": cx["chexbert_per_label_f1"],
        "chexbert_labels": cx["chexbert_labels"],
    }

def save_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

# =========================
# Testing helpers (optional, useful for pytest)
# =========================

def _reset_metric_caches() -> None:
    """Reset cached scorers (useful in tests to ensure clean state)."""
    global _CHEX_SCORER, _RG_SCORER
    _CHEX_SCORER = None
    _RG_SCORER = None

# =========================
# Example usage (commented)
# =========================
# def main():
#     generated_reports = [
#         "The lungs are clear. No pleural effusion.",
#         "There is mild cardiomegaly."
#     ]
#     original_reports  = [
#         "Lungs appear clear with no effusion.",
#         "Mild enlargement of the heart is noted."
#     ]
#     all_metrics = evaluate_all_metrics(generated_reports, original_reports)
#     for metric, scores in all_metrics.items():
#         print(f"{metric}: {scores}")
#     save_metrics_to_json(all_metrics, "lstm-vs-gpt/all_metrics.json")
#
# if __name__ == "__main__":
#     main()
