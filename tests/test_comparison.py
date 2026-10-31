import numpy as np
import pandas as pd
import pytest

from proyecto.models.candidates import make_logistic
from proyecto.models.comparison import (
    compare_pair_on_new_folds,
    confirm_on_new_folds,
    control_table,
    improvement_table,
    measure_variants,
    pairwise_table,
)
from proyecto.models.evaluate import repeated_folds
from tests.test_pipeline import _clients

REF = "referencia"


def _results():
    """Resultados escritos a mano: la variante gana siempre ~0,02 de AUC en un modelo y empeora en el otro."""
    ref = pd.DataFrame({"roc_auc": [0.70, 0.71, 0.69, 0.70, 0.72], "avg_precision": [0.50] * 5, "penalty": [0.60] * 5})
    better = ref.assign(roc_auc=ref.roc_auc + [0.020, 0.022, 0.018, 0.021, 0.019], avg_precision=0.52, penalty=0.58)
    worse = ref.assign(roc_auc=ref.roc_auc - 0.01)
    return {"logistica": {REF: ref, "mejora": better, "empeora": worse}, "lgbm": {REF: ref, "mejora": ref, "empeora": worse}}


VARIANTS = {REF: ((), ()), "mejora": ((), ()), "empeora": ((), ())}
MODELS = {"logistica": None, "lgbm": None}


def test_improvement_table_computes_the_gain_and_applies_both_decision_rules():
    table, comparisons = improvement_table(_results(), VARIANTS, MODELS, REF)
    assert table.loc["mejora", "logistica: mejora"] == pytest.approx(0.02)
    assert table.loc["mejora", "logistica: pliegues"] == "5/5"
    assert table.loc["mejora", "decisión (pareada)"] == "conservar"      # gana siempre en la logística y no empeora lgbm
    assert table.loc["empeora", "decisión (pareada)"] == "descartar"
    assert table.loc[REF, "decisión (pareada)"] == "—"
    assert comparisons["mejora"]["logistica"]["folds_improved"] == 5


def test_control_table_reports_differences_with_the_reference_where_positive_is_better():
    table = control_table(_results(), VARIANTS, MODELS, REF)
    row = table.loc[("logistica", "mejora")]
    assert row["Δ AUC-PR"] == pytest.approx(0.02)
    assert row["Δ penalización (positivo = mejor)"] == pytest.approx(0.02)       # 0,60 − 0,58: bajó, es mejor
    assert table.loc[("lgbm", REF), "Δ AUC-PR"] == 0


def test_pairwise_table_puts_the_model_with_the_higher_mean_first():
    per_fold = {"a": pd.DataFrame({"roc_auc": [0.70, 0.71, 0.72]}), "b": pd.DataFrame({"roc_auc": [0.75, 0.76, 0.75]})}
    table = pairwise_table(per_fold, [("a", "b")])
    assert table.loc[0, "mejor"] == "b" and table.loc[0, "contra"] == "a"
    assert table.loc[0, "pliegues que gana"] == "3 de 3"
    assert table.loc[0, "diferencia media de AUC"] == pytest.approx((0.05 + 0.05 + 0.03) / 3)


def test_measure_variants_returns_one_row_per_fold_for_every_variant_and_model():
    X, y = _clients(300)
    folds = repeated_folds(X, y, n_splits=3, n_repeats=1)
    variants = {REF: ((), ()), "con atraso": (("n_delayed_months",), ()), "sin SEX": ((), ("SEX",))}
    results = measure_variants({"Logística": make_logistic}, variants, X, y, folds)
    assert set(results["Logística"]) == set(variants)
    assert all(len(df) == 3 and df["roc_auc"].between(0.5, 1).all() for df in results["Logística"].values())


def test_confirm_on_new_folds_returns_the_decision_for_each_seed():
    X, y = _clients(300)
    out = confirm_on_new_folds(X, y, {"Logística": make_logistic}, ("n_delayed_months",), seeds=(7, 99))
    assert set(out) == {7, 99}
    per_model, keep_paired, keep_original = out[7]
    assert set(per_model["Logística"]) >= {"gain", "paired_gain", "gain_over_noise", "folds_improved", "n_folds"}
    assert isinstance(keep_paired, bool) and isinstance(keep_original, bool)
    assert per_model["Logística"]["n_folds"] == 15


def test_compare_pair_on_new_folds_compares_variant_against_reference_in_each_seed():
    X, y = _clients(300)
    out = compare_pair_on_new_folds(X, y, lambda: make_logistic(), lambda: make_logistic(("n_delayed_months",)), seeds=(7,))
    assert set(out) == {7} and out[7]["n_folds"] == 15


class _Constant:
    """Modelo sin información: da la misma probabilidad a todos (AUC 0,5)."""
    fitted_on: list = []

    def fit(self, X, y):
        _Constant.fitted_on.append(set(X.index))
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class _ByPay0(_Constant):
    """Modelo informativo: ordena por PAY_0 (en los datos de prueba, más atraso significa más probabilidad de incumplir)."""

    def predict_proba(self, X):
        p = (X["PAY_0"].to_numpy(dtype=float) + 2) / 12
        return np.column_stack([1 - p, p])


def _factory(calls=None):
    def factory(features, drop):
        if calls is not None:
            calls.append((tuple(features), tuple(drop)))
        return _ByPay0() if "n_delayed_months" in features else _Constant()
    return factory


def test_measure_variants_builds_each_variant_with_its_own_features_and_drop():
    X, y = _clients(300)
    calls = []
    variants = {REF: ((), ()), "con atraso": (("n_delayed_months",), ()), "sin SEX": ((), ("SEX",))}
    results = measure_variants({"m": _factory(calls)}, variants, X, y, repeated_folds(X, y, n_splits=3, n_repeats=1))
    assert set(calls) == {((), ()), (("n_delayed_months",), ()), ((), ("SEX",))}
    assert results["m"]["con atraso"]["roc_auc"].gt(0.6).all()              # esta variante sí recibió el modelo informativo
    assert (results["m"][REF]["roc_auc"] == 0.5).all() and (results["m"]["sin SEX"]["roc_auc"] == 0.5).all()


def test_confirm_on_new_folds_has_the_right_sign_and_uses_the_folds_of_each_seed():
    X, y = _clients(300)
    _Constant.fitted_on = []
    out = confirm_on_new_folds(X, y, {"m": _factory()}, ("n_delayed_months",), seeds=(7,))
    per_model, keep_paired, _ = out[7]
    assert per_model["m"]["gain"] > 0.1 and per_model["m"]["folds_improved"] == 15 and keep_paired
    expected = repeated_folds(X, y, n_splits=5, n_repeats=3, seed=7)
    for fitted, (train_idx, _) in zip(_Constant.fitted_on[:15], expected):     # las 15 primeras: la referencia
        assert fitted == set(X.index[train_idx])                                 # se ajusta solo con el entrenamiento del pliegue
    no_change = confirm_on_new_folds(X, y, {"m": _factory()}, (), seeds=(7,))
    assert no_change[7][0]["m"]["gain"] == 0                                    # sin variables nuevas, referencia = variante


def test_compare_pair_on_new_folds_does_not_swap_reference_and_variant():
    X, y = _clients(300)
    better = compare_pair_on_new_folds(X, y, _Constant, _ByPay0, seeds=(7,))[7]
    worse = compare_pair_on_new_folds(X, y, _ByPay0, _Constant, seeds=(7,))[7]
    assert better["gain"] > 0.1 and better["folds_improved"] == 15
    assert worse["gain"] < -0.1 and worse["folds_improved"] == 0


def test_improvement_table_also_applies_the_original_rule_in_its_own_column():
    """Cada pliegue tiene su dificultad (AUC entre 0,60 y 0,80), pero la variante gana siempre ~0,01: la regla pareada la
    conserva y la original (mejora / ruido) no."""
    ref = pd.DataFrame({"roc_auc": [0.60, 0.70, 0.80, 0.65, 0.75], "avg_precision": 0.5, "penalty": 0.6})
    var = ref.assign(roc_auc=ref.roc_auc + [0.010, 0.012, 0.009, 0.011, 0.010])
    results = {"m": {REF: ref, "v": var}}
    table, _ = improvement_table(results, {REF: ((), ()), "v": ((), ())}, {"m": None}, REF)
    assert table.loc["v", "decisión (pareada)"] == "conservar" and table.loc["v", "decisión (original)"] == "descartar"
