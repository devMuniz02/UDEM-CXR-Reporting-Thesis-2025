from __future__ import annotations
from typing import Union, List, Tuple, Dict, Any
import os
import pathlib
import numpy as np

# =========================
# Classic/NLP metrics
# =========================
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Optional: BERTScore (domain-aware)
try:
    from bert_score import score as bertscore_score
except Exception:
    bertscore_score = None

TextLike = Union[str, List[str]]

# -------------------------
# Utilities
# -------------------------
def _as_lists(generated: TextLike, original: TextLike) -> Tuple[List[str], List[str]]:
    """Coerce inputs to aligned lists of the same length (min of the two)."""
    if isinstance(generated, str): generated = [generated]
    if isinstance(original, str):  original  = [original]
    n = min(len(generated), len(original))
    return generated[:n], original[:n]

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
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
    """ROUGE-L (F1) per pair."""
    gen, ref = _as_lists(generated, original)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [float(scorer.score(r, g)['rougeL'].fmeasure) for g, r in zip(gen, ref)]

def cosine_sim_score(generated: TextLike, original: TextLike) -> List[float]:
    gen, ref = _as_lists(generated, original)
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
    except Exception:
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
    Wraps F1CheXbert to expose:
      - chexbert_f1: mean per-pair micro-F1 (F1CheXbert 'accuracy')
      - chexbert_f1_micro: dataset-level micro-F1 from classification report
      - chexbert_f1_macro: dataset-level macro-F1 over 14 labels
      - chexbert_f1_micro_5 / chexbert_f1_macro_5: top-5 subset if provided by the lib
      - chexbert_per_pair_micro: list of per-pair micro-F1
      - chexbert_per_label_f1: length-14 vector (order = _CHEXPERT_14)
    """
    gen, ref = _as_lists(generated, original)
    try:
        from f1chexbert import F1CheXbert
        scorer = F1CheXbert()  # requires chexbert.pth in its cache path
        accuracy, accuracy_not_averaged, class_report, class_report_5 = scorer(hyps=gen, refs=ref)

        micro_f1   = float(class_report.get("micro avg", {}).get("f1-score", 0.0))
        macro_f1   = float(class_report.get("macro avg", {}).get("f1-score", 0.0))
        micro_f1_5 = float(class_report_5.get("micro avg", {}).get("f1-score", 0.0))
        macro_f1_5 = float(class_report_5.get("macro avg", {}).get("f1-score", 0.0))
        per_label  = [float(class_report.get(lbl, {}).get("f1-score", 0.0)) for lbl in _CHEXPERT_14]

        return {
            "chexbert_f1": float(accuracy),
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
    except Exception as e:
        print("[CheXbert] unavailable:", repr(e))

    # Safe empty return on failure
    return {
        "chexbert_f1": None,
        "chexbert_f1_micro": None,
        "chexbert_f1_macro": None,
        "chexbert_f1_micro_5": None,
        "chexbert_f1_macro_5": None,
        "chexbert_per_pair_micro": [],
        "chexbert_per_label_f1": [],
        "chexbert_labels": _CHEXPERT_14,
    }

# =========================
# RadGraph (entity/relations F1 via F1RadGraph)
# =========================
import contextlib
import io

def radgraph_metric(generated: TextLike, original: TextLike, which: str = "RG_ER") -> List[float]:
    """
    Per-pair RadGraph F1 using the official scorer.
    which ∈ {'RG_E','RG_ER','RG_bar_ER'}; default 'RG_ER'.
    Robust to versions that append a trailing batch-mean entry.
    Suppresses device/model_type messages from RadGraph.
    """
    gen, ref = _as_lists(generated, original)

    # Honor lightweight mode if set
    if os.environ.get("RADGRAPH_DISABLE_SCI", "").lower() in {"1", "true", "yes"}:
        os.environ["DISABLE_RADGRAPH_SCISPACY"] = "1"

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from radgraph import F1RadGraph
            idx = {"RG_E": 0, "RG_ER": 1, "RG_bar_ER": 2}[which]
            f1rg = F1RadGraph(reward_level="all")
            mean_reward, reward_list, *_ = f1rg(hyps=gen, refs=ref)

        # Normalize to a flat list of floats
        scores = []
        for t in reward_list:
            if isinstance(t, (list, tuple)):
                scores.append(float(t[idx]))
            else:
                scores.append(float(t))
        if len(scores) > len(gen):
            scores = scores[:len(gen)]
        return scores

    except Exception as e:
        print("[RadGraph] NOT usable:", repr(e))
        return []
# =========================
# Master wrapper
# =========================
def evaluate_all_metrics(generated: TextLike, original: TextLike, evaluation_mode: str = "default") -> Dict[str, Any]:
    gen, ref = _as_lists(generated, original)

    cx = chexbert_metrics(gen, ref)
    bs = bertscore_metric(gen, ref)

    # Always compute RadGraph variants like before
    rg_e  = radgraph_metric(gen, ref, which="RG_E")
    rg_er = radgraph_metric(gen, ref, which="RG_ER")

    if evaluation_mode == "CheXagent":
        return {
            # CheXbert-focused (as in your figure/papers)
            "chexbert_f1": cx["chexbert_f1"],
            "chexbert_f1_micro": cx["chexbert_f1_micro"],
            "chexbert_f1_macro": cx["chexbert_f1_macro"],
            "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],
            "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],
            "bertscore_f1": bs["f1"],
            "rouge_l": rouge_l_score(gen, ref),
            # RadGraph F1 (like you had before)
            "radgraph_f1_RG_E": rg_e,
            "radgraph_f1_RG_ER": rg_er,
        }

    return {
        # classic
        "bleu": bleu_score(gen, ref),
        "rouge_l": rouge_l_score(gen, ref),          # ROUGE-L (F1)
        "cosine_similarity": cosine_sim_score(gen, ref),
        "meteor": meteor_score_batch(gen, ref),

        # bertscore (PubMedBERT by default)
        "bertscore_f1": bs["f1"],

        # chexbert variants
        "chexbert_f1": cx["chexbert_f1"],                        # mean per-pair micro-F1
        "chexbert_f1_micro": cx["chexbert_f1_micro"],            # dataset micro-F1
        "chexbert_f1_macro": cx["chexbert_f1_macro"],            # dataset macro-F1
        "chexbert_f1_micro_5": cx["chexbert_f1_micro_5"],        # dataset micro-F1 (top-5)
        "chexbert_f1_macro_5": cx["chexbert_f1_macro_5"],        # dataset macro-F1 (top-5)
        "chexbert_per_pair_micro": cx["chexbert_per_pair_micro"],
        "chexbert_per_label_f1": cx["chexbert_per_label_f1"],
        "chexbert_labels": cx["chexbert_labels"],

        # RadGraph F1 (like before)
        "radgraph_f1_RG_E": rg_e,
        "radgraph_f1_RG_ER": rg_er,
    }

# =========================
# Example usage
# =========================
def main():
    generated_reports = [
        "The lungs are clear. No pleural effusion.",
        "There is mild cardiomegaly."
    ]
    original_reports  = [
        "Lungs appear clear with no effusion.",
        "Mild enlargement of the heart is noted."
    ]

    all_metrics = evaluate_all_metrics(generated_reports, original_reports)  # or evaluation_mode="CheXagent"
    for metric, scores in all_metrics.items():
        print(f"{metric}: {scores}")

if __name__ == "__main__":
    main()
