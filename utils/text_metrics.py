
"""
Text metrics utilities for chest X-ray report evaluation.
Includes BLEU, ROUGE, METEOR, cosine similarity, BERTScore, CheXbert, and RadGraph metrics.
"""


# Standard library imports
from __future__ import annotations
import os
import pathlib

# Typing imports
from typing import Union, List, Tuple, Dict, Any

# Third-party imports
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
from radgraph import F1RadGraph

# CheXbert imports
from f1chexbert import F1CheXbert

TextLike = Union[str, List[str]]

def _as_lists(generated: TextLike, original: TextLike) -> Tuple[List[str], List[str]]:
    """
    Coerce inputs to aligned lists of the same length (min of the two).
    Args:
        generated (TextLike): Generated text(s).
        original (TextLike): Reference text(s).
    Returns:
        Tuple[List[str], List[str]]: Aligned lists of generated and reference texts.
    """
    if isinstance(generated, str):
        generated = [generated]
    if isinstance(original, str):
        original = [original]
    n = min(len(generated), len(original))
    return generated[:n], original[:n]

def _has_cuda() -> bool:
    """
    Check if CUDA is available for torch.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        return cuda.is_available()
    except ImportError:
        return False

# =========================
# Classic text metrics
# =========================
def bleu_score(generated: TextLike, original: TextLike) -> List[float]:
    """
    Compute BLEU score for each generated/reference pair.
    Args:
        generated (TextLike): Generated text(s).
        original (TextLike): Reference text(s).
    Returns:
        List[float]: BLEU scores per pair.
    """
    gen, ref = _as_lists(generated, original)
    smoothie = SmoothingFunction().method1
    return [float(sentence_bleu([r.split()], g.split(), smoothing_function=smoothie))
            for g, r in zip(gen, ref)]

def rouge_l_score(generated: TextLike, original: TextLike) -> List[float]:
    """
    Compute ROUGE-L (F1) score for each generated/reference pair.
    Args:
        generated (TextLike): Generated text(s).
        original (TextLike): Reference text(s).
    Returns:
        List[float]: ROUGE-L F1 scores per pair.
    """
    gen, ref = _as_lists(generated, original)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [float(scorer.score(r, g)['rougeL'].fmeasure) for g, r in zip(gen, ref)]

def cosine_sim_score(generated: TextLike, original: TextLike) -> List[float]:
    """
    Compute cosine similarity score for each generated/reference pair using TF-IDF vectors.
    Args:
        generated (TextLike): Generated text(s).
        original (TextLike): Reference text(s).
    Returns:
        List[float]: Cosine similarity scores per pair.
    """
    gen, ref = _as_lists(generated, original)
    vec = TfidfVectorizer().fit(gen + ref)
    out = []
    for g, r in zip(gen, ref):
        tfidf = vec.transform([g, r])
        out.append(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]))
    return out

def meteor_score_batch(generated: TextLike, original: TextLike) -> List[float]:
    """
    Compute METEOR score for each generated/reference pair.
    Args:
        generated (TextLike): Generated text(s).
        original (TextLike): Reference text(s).
    Returns:
        List[float]: METEOR scores per pair.
    """
    gen, ref = _as_lists(generated, original)
    return [float(meteor_score([r.split()], g.split())) for g, r in zip(gen, ref)]

# =========================
# BERTScore (biomedical default)
# =========================
def bertscore_metric(generated: TextLike, original: TextLike,
                     model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                     device: str | None = None) -> Dict[str, List[float]]:
    """
    Returns dict with lists: {'p': [...], 'r': [...], 'f1': [...]}
    """
    if bertscore_score is None:
        return {"p": [], "r": [], "f1": []}
    gen, ref = _as_lists(generated, original)
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
# Label order used for per-label reporting
_CHEXPERT_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

def chexbert_metrics(generated: TextLike, original: TextLike) -> Dict[str, Any]:
    """
    Compute CheXbert metrics for generated/reference report pairs.
    Wraps F1CheXbert to expose:
        - chexbert_f1_weighted: mean per-pair weighted F1
        - chexbert_f1_micro: dataset-level micro-F1
        - chexbert_f1_macro: dataset-level macro-F1
        - chexbert_f1_micro_5 / chexbert_f1_macro_5: top-5 subset if provided
        - chexbert_per_pair_micro: list of per-pair micro-F1
        - chexbert_per_label_f1: length-14 vector (order = _CHEXPERT_14)
    Args:
            generated (TextLike): Generated report(s).
            original (TextLike): Reference report(s).
    Returns:
            Dict[str, Any]: CheXbert metrics and per-label F1 scores.
    """
    gen, ref = _as_lists(generated, original)
    try:
        scorer = F1CheXbert()  # requires chexbert.pth in its cache path
        # accuracy, accuracy_not_averaged, class_report, class_report_5
        _, accuracy_not_averaged, class_report, class_report_5 = scorer(hyps=gen, refs=ref)

        weighted_f1 = float(class_report.get("weighted avg", {}).get("f1-score", 0.0))
        micro_f1   = float(class_report.get("micro avg", {}).get("f1-score", 0.0))
        macro_f1   = float(class_report.get("macro avg", {}).get("f1-score", 0.0))
        micro_f1_5 = float(class_report_5.get("micro avg", {}).get("f1-score", 0.0))
        macro_f1_5 = float(class_report_5.get("macro avg", {}).get("f1-score", 0.0))
        per_label  = [float(class_report.get(lbl, {}).get("f1-score", 0.0)) for lbl in _CHEXPERT_14]

        return {
            "chexbert_f1_weighted": weighted_f1,
            "chexbert_f1_micro": micro_f1,
            "chexbert_f1_macro": macro_f1,
            "chexbert_f1_micro_5": micro_f1_5,
            "chexbert_f1_macro_5": macro_f1_5,
            "chexbert_per_pair_micro": [float(x) for x in list(accuracy_not_averaged)],
            "chexbert_per_label_f1": per_label,
            "chexbert_labels": _CHEXPERT_14,
        }
    except FileNotFoundError as e:
        missing_path = getattr(e, "filename", "") or str(
            pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home()/"AppData"/"Local"))
            / "chexbert" / "chexbert" / "Cache" / "chexbert.pth"
        )
        print("[CheXbert] Weights missing at:", missing_path)
        print("→ Place 'chexbert.pth' there and re-run.")
    except ImportError as e:
        print("[CheXbert] unavailable (import error):", repr(e))
    except AttributeError as e:
        print("[CheXbert] unavailable (attribute error):", repr(e))

    # Safe empty return on failure
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

def radgraph_metric(generated: TextLike, original: TextLike) -> List[float]:
    """
    Compute RadGraph entity/relation F1 metrics for generated/reference report pairs.
    Args:
        generated (TextLike): Generated report(s).
        original (TextLike): Reference report(s).
    Returns:
        List[float]: Tuple of (entity F1, entity+relation F1, bar entity+relation F1).
    """
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    hyps = generated
    refs = original
    # mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists
    mean_reward, *_ = f1radgraph(hyps=hyps, refs=refs)

    rg_e, rg_er, rg_bar_er = mean_reward

    return float(rg_e), float(rg_er), float(rg_bar_er)

# =========================
# All metrics evaluation
# =========================

def evaluate_all_metrics(generated: TextLike,
                         original: TextLike,
                         evaluation_mode: str = "CheXagent") -> Dict[str, Any]:
    """
    Evaluate all text metrics for generated/reference report pairs.
    Args:
        generated (TextLike): Generated report(s).
        original (TextLike): Reference report(s).
        evaluation_mode (str): Evaluation mode ('default' or 'CheXagent').
    Returns:
        Dict[str, Any]: Dictionary of all computed metrics.
    """
    gen, ref = _as_lists(generated, original)

    cx = chexbert_metrics(gen, ref)
    bs = bertscore_metric(gen, ref)

    # Always compute RadGraph variants like before
    rg_e, rg_er, _ = radgraph_metric(gen, ref)

    if evaluation_mode == "CheXagent":
        return {
            # CheXbert-focused (as in your figure/papers)
            "chexbert_f1_weighted": cx["chexbert_f1_weighted"],
            "chexbert_f1_micro": cx["chexbert_f1_micro"],
            "chexbert_f1_macro": cx["chexbert_f1_macro"],
            "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],
            "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],
            "bertscore_f1": bs["f1"],
            # RadGraph F1 (like you had before)
            "radgraph_f1_RG_E": rg_e,
            "radgraph_f1_RG_ER": rg_er,
            "rouge_l": rouge_l_score(gen, ref), 
        }

    return {
        # classic
        "bleu": bleu_score(gen, ref),
        "rouge_l": rouge_l_score(gen, ref),          # ROUGE-L (F1)
        "cosine_similarity": cosine_sim_score(gen, ref),
        "meteor": meteor_score_batch(gen, ref),
        # bertscore (PubMedBERT by default)
        "bertscore_f1": bs["f1"],
        # RadGraph F1
        "radgraph_f1_RG_E": rg_e,
        "radgraph_f1_RG_ER": rg_er,
        # chexbert variants
        "chexbert_f1_weighted": cx["chexbert_f1_weighted"],      # mean per-pair micro-F1
        "chexbert_f1_micro": cx["chexbert_f1_micro"],            # dataset micro-F1
        "chexbert_f1_macro": cx["chexbert_f1_macro"],            # dataset macro-F1
        "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],        # dataset micro-F1 (top-5)
        "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],        # dataset macro-F1 (top-5)
        "chexbert_per_pair_micro": cx["chexbert_per_pair_micro"],
        "chexbert_per_label_f1": cx["chexbert_per_label_f1"],
        "chexbert_labels": cx["chexbert_labels"],
    }

# =========================
# Example usage
# =========================
def main():
    """
    Example usage for evaluating metrics on sample reports.
    """
    generated_reports = [
        "The lungs are clear. No pleural effusion.",
        "There is mild cardiomegaly."
    ]
    original_reports  = [
        "Lungs appear clear with no effusion.",
        "Mild enlargement of the heart is noted."
    ]

    all_metrics = evaluate_all_metrics(generated_reports, original_reports)
    for metric, scores in all_metrics.items():
        print(f"{metric}: {scores}")

if __name__ == "__main__":
    main()
