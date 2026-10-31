import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from proyecto.models.evaluate import (
    THEORETICAL_THRESHOLD,
    decide,
    improvement_over_noise,
    relative_penalty,
    repeated_folds,
    score_folds,
)
from proyecto.models.train import build_pipeline
from tests.test_pipeline import _clients


def test_relative_penalty_weights_false_negatives_five_times():
    y = [1, 1, 0, 0]
    proba = [0.9, 0.0, 0.9, 0.0]          # 1 acierto, 1 falso negativo, 1 falsa alarma, 1 acierto
    assert relative_penalty(y, proba) == (5 * 1 + 1) / 4


def test_relative_penalty_flags_the_client_at_exactly_the_threshold():
    assert relative_penalty([1], [THEORETICAL_THRESHOLD]) == 0.0      # se marca: no es falso negativo
    assert relative_penalty([1], [THEORETICAL_THRESHOLD - 1e-9]) == 5.0


def test_theoretical_threshold_is_one_sixth_for_five_to_one():
    assert THEORETICAL_THRESHOLD == pytest.approx(1 / 6)


def test_repeated_folds_are_stratified_and_do_not_share_rows_within_a_fold():
    X, y = _clients()
    folds = repeated_folds(X, y, n_splits=5, n_repeats=2)
    assert len(folds) == 10
    for train_idx, valid_idx in folds:
        assert set(train_idx).isdisjoint(valid_idx)
        assert abs(y[valid_idx].mean() - y.mean()) < 0.05


def test_score_folds_returns_one_row_per_fold_with_valid_metrics():
    X, y = _clients()
    folds = repeated_folds(X, y, n_splits=3, n_repeats=1)
    scores = score_folds(lambda: build_pipeline(LogisticRegression(max_iter=500), linear=True), X, y, folds)
    assert len(scores) == 3
    assert scores["roc_auc"].between(0.5, 1).all()        # PAY_0 sí informa el objetivo sintético
    assert (scores["penalty"] >= 0).all()


def test_improvement_over_noise_with_known_numbers():
    ref = [0.70, 0.71, 0.69, 0.70, 0.70]
    var = [0.72, 0.73, 0.71, 0.72, 0.72]
    result = improvement_over_noise(ref, var)
    assert result["gain"] == pytest.approx(0.02)
    assert result["noise"] == pytest.approx(np.std(var, ddof=1))
    assert result["gain_over_noise"] == pytest.approx(0.02 / np.std(var, ddof=1))
    assert result["folds_improved"] == 5


def test_improvement_over_noise_flips_the_sign_when_lower_is_better():
    result = improvement_over_noise([0.5, 0.5, 0.5], [0.4, 0.45, 0.35], higher_is_better=False)
    assert result["gain"] > 0 and result["folds_improved"] == 3


def _result(gain, ratio, share_ok, total=15, paired=None):
    """`ratio` es el cociente original (gain_over_noise); `paired` el pareado (por defecto igual al original)."""
    return {"gain": gain, "noise": 0.01, "gain_over_noise": ratio,
            "paired_gain": ratio if paired is None else paired, "folds_improved": share_ok, "n_folds": total}


def test_decide_keeps_an_idea_that_passes_in_one_model_and_does_not_hurt_the_other():
    keep, why = decide({"lgbm": _result(0.002, 0.4, 12), "logistica": _result(0.009, 1.5, 15)})
    assert keep and "logistica" in why


def test_decide_rejects_when_it_hurts_any_model_even_if_it_passes_in_another():
    keep, _ = decide({"lgbm": _result(-0.001, -0.1, 5), "logistica": _result(0.009, 1.5, 15)})
    assert not keep


def test_decide_rejects_a_ratio_above_one_that_only_wins_in_few_folds():
    keep, _ = decide({"lgbm": _result(0.002, 1.2, 10), "logistica": _result(0.0, 0.0, 7)})
    assert not keep


def test_decide_rejects_noise():
    keep, why = decide({"lgbm": _result(0.001, 0.1, 9), "logistica": _result(0.001, 0.2, 9)})
    assert not keep and "ruido" in why


def test_paired_ratio_uses_the_spread_of_the_differences_not_of_the_variant():
    # Cada pliegue tiene su dificultad (AUC entre 0,60 y 0,80), pero la variante gana siempre ~0,01
    ref = np.array([0.60, 0.70, 0.80, 0.65, 0.75])
    var = ref + np.array([0.010, 0.012, 0.009, 0.011, 0.010])
    r = improvement_over_noise(ref, var)
    assert r["gain_over_noise"] < 1 < r["paired_gain"]
    assert r["paired_gain"] == pytest.approx(0.0104 / np.std([0.010, 0.012, 0.009, 0.011, 0.010], ddof=1))


def test_decide_with_the_original_ratio_rejects_what_the_paired_ratio_keeps():
    per_model = {"logistica": _result(0.009, 0.81, 15, paired=4.0), "lgbm": _result(0.002, 0.3, 14, paired=1.5)}
    assert decide(per_model)[0]                                          # regla vigente (pareada)
    assert not decide(per_model, key="gain_over_noise")[0]            # regla original


class _SpyModel:
    """Falso modelo que anota con qué filas se ajustó y sobre cuáles predijo."""

    def __init__(self):
        self.fit_index = None
        self.predict_index = None

    def fit(self, X, y):
        self.fit_index = set(X.index)
        return self

    def predict_proba(self, X):
        self.predict_index = set(X.index)
        return np.full((len(X), 2), 0.5)


def test_score_folds_fits_only_on_training_rows_and_scores_only_on_validation_rows():
    """Si el ajuste usara filas de validación (fuga), este test fallaría."""
    X, y = _clients()
    folds = repeated_folds(X, y, n_splits=3, n_repeats=1)
    spies = []

    def make():
        spies.append(_SpyModel())
        return spies[-1]

    score_folds(make, X, y, folds)
    assert len(spies) == len(folds)                      # un modelo nuevo por pliegue
    for spy, (train_idx, valid_idx) in zip(spies, folds):
        assert spy.fit_index == set(X.index[train_idx])
        assert spy.predict_index == set(X.index[valid_idx])
        assert spy.fit_index.isdisjoint(spy.predict_index)


def test_decide_at_the_exact_borders():
    # cociente exactamente 1 no es «mayor que 1»
    assert not decide({"m": _result(0.01, 1.0, 15)})[0]
    # 12 de 15 pliegues alcanzan el 80 %; 11 de 15 no
    assert decide({"m": _result(0.01, 1.5, 12)})[0]
    assert not decide({"m": _result(0.01, 1.5, 11)})[0]
    # una mejora promedio de exactamente 0 no cuenta como empeorar, pero tampoco pasa
    assert not decide({"m": _result(0.0, 0.0, 15)})[0]
