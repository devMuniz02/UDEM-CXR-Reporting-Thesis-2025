# test_text_metrics.py
# Pytest suite for text_metrics.py
# Assumes the implementation is saved as text_metrics.py in the same project.
# If your file has a different name, change the import line below.

import json
import numpy as np
import pytest

import utils.text_metrics as tm


# -------------------------
# Fixtures & tiny stubs
# -------------------------

@pytest.fixture
def sample_texts():
    gen = [
        "There is mild cardiomegaly with small pleural effusion.",
        "No acute cardiopulmonary process is identified.",
        "Lines and tubes are in expected position."
    ]
    ref = [
        "Mild enlargement of the cardiac silhouette and a small pleural effusion.",
        "No acute cardiopulmonary abnormality.",
        "Support devices are appropriately positioned."
    ]
    return gen, ref


class _TensorStub:
    """Minimal tensor-like stub to emulate bert-score return values."""
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def tolist(self):
        return self._arr.tolist()


class _F1CheXbertStub:
    """Callable stub emulating F1CheXbert inference & reports."""
    def __call__(self, hyps, refs):
        # accuracy: dataset-level? We'll return something plausible.
        accuracy = 0.75
        # per-pair micro-F1 list, one value per pair
        accuracy_not_averaged = [0.6 for _ in hyps]
        # class_report dict containing per-label and avg blocks
        labels = tm._CHEXPERT_14
        class_report = {
            "weighted avg": {"f1-score": 0.7},
            "micro avg": {"f1-score": 0.65},
            "macro avg": {"f1-score": 0.6},
        }
        # give each label an f1-score
        for lbl in labels:
            class_report[lbl] = {"f1-score": 0.5}
        # top-5 variant
        class_report_5 = {
            "micro avg": {"f1-score": 0.66},
            "macro avg": {"f1-score": 0.61},
        }
        return accuracy, accuracy_not_averaged, class_report, class_report_5


class _F1RadGraphStub:
    """Callable stub for RadGraph metric."""
    def __init__(self, reward_level="all", model_type="radgraph-xl"):
        self.reward_level = reward_level
        self.model_type = model_type

    def __call__(self, hyps, refs):
        # mean_reward is a triple (entity F1, entity+relation F1, bar entity+relation F1)
        mean_reward = (0.42, 0.36, 0.35)
        # The function actually returns 4 values; we can fill others with None
        return mean_reward, None, None, None


# ---------------------------------
# Unit tests for helper utilities
# ---------------------------------

def test_as_lists_str_and_list():
    g, r = tm._as_lists("hello world", ["hello", "world"])
    assert isinstance(g, list) and isinstance(r, list)
    assert len(g) == len(r) == 1
    assert g[0] == "hello world"
    assert r[0] == "hello"


def test_has_cuda_monkeypatched(monkeypatch):
    monkeypatch.setattr(tm.cuda, "is_available", lambda: True)
    assert tm._has_cuda() is True
    monkeypatch.setattr(tm.cuda, "is_available", lambda: False)
    assert tm._has_cuda() is False


# ---------------------------------
# Classic metrics (lightweight)
# ---------------------------------

def test_bleu_rouge_cosine_meteor_basic(monkeypatch, sample_texts):
    gen, ref = sample_texts

    # Some environments lack NLTK data for METEOR; stabilize via stub
    def _meteor_stub(references, hypothesis):
        # simple overlap fraction stub
        rset = set(references[0])
        hset = set(hypothesis)
        return len(rset & hset) / max(1, len(rset | hset))

    monkeypatch.setattr(tm, "meteor_score", _meteor_stub)

    bleu = tm.bleu_score(gen, ref)
    rouge = tm.rouge_l_score(gen, ref)
    cos = tm.cosine_sim_score(gen, ref)
    met = tm.meteor_score_batch(gen, ref)

    # Basic shape & bounds checks
    for arr in (bleu, rouge, cos, met):
        assert isinstance(arr, list)
        assert len(arr) == len(gen)
        for v in arr:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


# ---------------------------------
# BERTScore
# ---------------------------------

def test_bertscore_when_package_missing(monkeypatch, sample_texts):
    gen, ref = sample_texts
    # Simulate that import failed and was set to None in the module
    monkeypatch.setattr(tm, "bertscore_score", None)
    out = tm.bertscore_metric(gen, ref)
    assert out == {"p": [], "r": [], "f1": []}


def test_bertscore_with_stub(monkeypatch, sample_texts):
    gen, ref = sample_texts

    def _bert_stub(gen_, ref_, model_type=None, device=None, verbose=False):
        n = len(gen_)
        # Return three tensor-like objects
        return (_TensorStub([0.8] * n),
                _TensorStub([0.7] * n),
                _TensorStub([0.75] * n))

    monkeypatch.setattr(tm, "bertscore_score", _bert_stub)
    # Force device choice to avoid peeking CUDA
    out = tm.bertscore_metric(gen, ref, model="dummy/model", device="cpu")
    assert list(out.keys()) == ["p", "r", "f1"]
    assert len(out["f1"]) == len(gen)
    assert all(abs(v - 0.75) < 1e-9 for v in out["f1"])


# ---------------------------------
# CheXbert
# ---------------------------------

def test_chexbert_metrics_with_stub(monkeypatch, sample_texts):
    gen, ref = sample_texts
    monkeypatch.setattr(tm, "F1CheXbert", _F1CheXbertStub)
    out = tm.chexbert_metrics(gen, ref)

    # Check keys
    for k in [
        "chexbert_f1_weighted", "chexbert_f1_micro", "chexbert_f1_macro",
        "chexbert_f1_micro_5", "chexbert_f1_macro_5",
        "chexbert_per_pair_micro", "chexbert_per_label_f1", "chexbert_labels"
    ]:
        assert k in out

    assert out["chexbert_f1_weighted"] == pytest.approx(0.7)
    assert out["chexbert_f1_micro"] == pytest.approx(0.65)
    assert out["chexbert_f1_macro"] == pytest.approx(0.6)
    assert out["chexbert_f1_micro_5"] == pytest.approx(0.66)
    assert out["chexbert_f1_macro_5"] == pytest.approx(0.61)
    assert len(out["chexbert_per_pair_micro"]) == len(gen)
    assert len(out["chexbert_per_label_f1"]) == len(tm._CHEXPERT_14)
    assert out["chexbert_labels"] == tm._CHEXPERT_14


def test_chexbert_metrics_missing_weights(monkeypatch, sample_texts, capsys):
    gen, ref = sample_texts

    class _F1CheXbertMissing:
        def __init__(self):
            pass
        def __call__(self, hyps, refs):
            raise FileNotFoundError("chexbert.pth")

    monkeypatch.setattr(tm, "F1CheXbert", _F1CheXbertMissing)

    out = tm.chexbert_metrics(gen, ref)
    # All should be None or empty on safe fallback
    assert out["chexbert_f1_weighted"] is None
    assert out["chexbert_f1_micro"] is None
    assert out["chexbert_f1_macro"] is None
    assert out["chexbert_f1_micro_5"] is None
    assert out["chexbert_f1_macro_5"] is None
    assert out["chexbert_per_pair_micro"] == []
    assert out["chexbert_per_label_f1"] == []
    assert out["chexbert_labels"] == tm._CHEXPERT_14

    # Ensure helpful message was printed
    captured = capsys.readouterr()
    assert "CheXbert" in captured.out


# ---------------------------------
# RadGraph
# ---------------------------------

def test_radgraph_metric_with_stub(monkeypatch, sample_texts):
    gen, ref = sample_texts
    monkeypatch.setattr(tm, "F1RadGraph", _F1RadGraphStub)
    rg_e, rg_er, rg_bar = tm.radgraph_metric(gen, ref)
    assert 0.0 <= rg_e <= 1.0
    assert 0.0 <= rg_er <= 1.0
    assert 0.0 <= rg_bar <= 1.0
    assert (rg_e, rg_er, rg_bar) == pytest.approx((0.42, 0.36, 0.35))


# ---------------------------------
# evaluate_all_metrics
# ---------------------------------

def test_evaluate_all_metrics_default_mode(monkeypatch, sample_texts):
    gen, ref = sample_texts

    # Stabilize external dependencies with stubs
    monkeypatch.setattr(tm, "F1RadGraph", _F1RadGraphStub)
    monkeypatch.setattr(tm, "F1CheXbert", _F1CheXbertStub)

    def _bert_stub(gen_, ref_, model_type=None, device=None, verbose=False):
        n = len(gen_)
        return (_TensorStub([0.9] * n),
                _TensorStub([0.9] * n),
                _TensorStub([0.9] * n))
    monkeypatch.setattr(tm, "bertscore_score", _bert_stub)

    # Also stub meteor to avoid nltk data issues
    monkeypatch.setattr(tm, "meteor_score", lambda refs, hyp: 0.5)

    out = tm.evaluate_all_metrics(gen, ref, evaluation_mode="default")

    # presence of expected keys
    for k in [
        "bleu", "rouge_l", "cosine_similarity", "meteor",
        "bertscore_f1",
        "radgraph_f1_RG_E", "radgraph_f1_RG_ER",
        "chexbert_f1_weighted", "chexbert_f1_micro", "chexbert_f1_macro",
        "chexbert_f1_micro_5", "chexbert_f1_macro_5",
        "chexbert_per_pair_micro", "chexbert_per_label_f1", "chexbert_labels"
    ]:
        assert k in out

    # shape checks
    assert len(out["bleu"]) == len(gen)
    assert len(out["rouge_l"]) == len(gen)
    assert len(out["cosine_similarity"]) == len(gen)
    assert len(out["meteor"]) == len(gen)
    assert len(out["bertscore_f1"]) == len(gen)
    assert isinstance(out["radgraph_f1_RG_E"], float)
    assert isinstance(out["radgraph_f1_RG_ER"], float)
    assert len(out["chexbert_per_label_f1"]) == len(tm._CHEXPERT_14)


def test_evaluate_all_metrics_chexagent_mode(monkeypatch, sample_texts):
    gen, ref = sample_texts

    monkeypatch.setattr(tm, "F1RadGraph", _F1RadGraphStub)
    monkeypatch.setattr(tm, "F1CheXbert", _F1CheXbertStub)
    monkeypatch.setattr(tm, "bertscore_score", None)  # not used in this mode

    out = tm.evaluate_all_metrics(gen, ref, evaluation_mode="CheXagent")

    # Only the focused subset should be present
    expected_keys = {
        "chexbert_f1_weighted",
        "chexbert_f1_micro",
        "chexbert_f1_macro",
        "chexbert_f1_micro_5",
        "chexbert_f1_macro_5",
        "radgraph_f1_RG_E",
        "radgraph_f1_RG_ER",
    }
    assert set(out.keys()) == expected_keys


# ---------------------------------
# save_metrics_to_json
# ---------------------------------

def test_save_metrics_to_json_roundtrip(tmp_path):
    metrics = {
        "a": [1.0, 0.5],
        "b": {"x": 3, "y": 4},
        "c": "ok",
    }
    fp = tmp_path / "metrics.json"
    tm.save_metrics_to_json(metrics, str(fp))
    assert fp.exists()
    with open(fp, "r") as f:
        loaded = json.load(f)
    assert loaded == metrics
