import numpy as np
import pandas as pd
import pytest

from proyecto.models.tournament import (
    best_by_scenario,
    calibration_by_bin,
    calibration_summary,
    is_baseline,
    rank_positions,
    summarize_by_model,
    technical_table,
    volume_per_thousand,
)


def _fold_frame(auc, **extra):
    n = len(auc)
    base = {"roc_auc": auc, "avg_precision": [0.5] * n, "balanced_accuracy": [0.6] * n, "f1_macro": [0.6] * n,
            "precision_1": [0.4] * n, "recall_0": [0.7] * n, "recall_1": [0.7] * n, "flagged_share": [0.4] * n,
            "penalty_2": [0.3] * n, "penalty": [0.5] * n, "penalty_10": [0.7] * n, "log_loss": [0.4] * n, "brier": [0.13] * n,
            "mean_proba": [0.22] * n, "recall_top_5": [0.17] * n, "recall_top_10": [0.3] * n, "recall_top": [0.5] * n,
            "lift_top": [2.5] * n, "fit_seconds": [0.2] * n}
    base.update(extra)
    return pd.DataFrame(base)


PER_FOLD = {"modelo A": _fold_frame([0.78, 0.79, 0.80]), "línea base: siempre paga": _fold_frame([0.5, 0.5, 0.5]),
            "modelo B": _fold_frame([0.70, 0.72, 0.74])}


def test_summarize_by_model_sorts_by_the_metric_and_reports_the_spread():
    table = summarize_by_model(PER_FOLD)
    assert list(table.index) == ["modelo A", "modelo B", "línea base: siempre paga"]
    assert table.loc["modelo A", "ROC AUC (media)"] == pytest.approx(0.79)
    assert table.loc["modelo A", "± entre pliegues"] == pytest.approx(0.01)


def test_technical_table_renames_columns_and_blanks_the_metrics_a_rule_cannot_have():
    table = technical_table(PER_FOLD, order=["modelo A", "línea base: siempre paga"])
    assert list(table.index) == ["modelo A", "línea base: siempre paga"]
    assert table.loc["modelo A", "penalización 5:1"] == pytest.approx(0.5)
    assert np.isnan(table.loc["línea base: siempre paga", "log loss"])        # una regla de 0 y 1 no tiene log loss
    assert table.loc["modelo A", "log loss"] == pytest.approx(0.4)


def test_rank_positions_orders_each_metric_in_the_right_direction_and_keeps_missing_values():
    table = pd.DataFrame({"AUC": [0.8, 0.7, 0.9], "penalización 5:1": [0.5, 0.6, np.nan]}, index=["a", "b", "c"])
    ranks = rank_positions(table, ["AUC", "penalización 5:1"])
    assert ranks["AUC"].tolist() == [2, 3, 1]                  # mayor AUC es mejor
    assert ranks["penalización 5:1"].tolist()[:2] == [1, 2]    # menor penalización es mejor
    assert pd.isna(ranks.loc["c", "penalización 5:1"])


def test_best_by_scenario_picks_the_lowest_penalty_and_the_highest_recall():
    table = pd.DataFrame({"penalización 5:1": [0.55, 0.50], "recall al revisar 20 %": [0.51, 0.52]}, index=["a", "b"])
    best = best_by_scenario(table, ["penalización 5:1", "recall al revisar 20 %"], ["a", "b"])
    assert best["penalización 5:1"] == ("b", 0.50) and best["recall al revisar 20 %"] == ("b", 0.52)


def test_volume_per_thousand_with_hand_computed_numbers():
    table = pd.DataFrame({"% marcado": [0.4], "recall (incumple)": [0.75], "penalización 5:1": [0.5],
                          "recall al revisar 20 %": [0.5]}, index=["m"])
    v = volume_per_thousand(table, positive_rate=0.2).loc["m"]
    assert v["se marcan para revisar"] == pytest.approx(400)
    assert v["incumplidores detectados"] == pytest.approx(150)
    assert v["incumplidores que se escapan"] == pytest.approx(50)
    assert v["falsas alarmas"] == pytest.approx(250)
    assert v["penalización total (5:1)"] == pytest.approx(500)
    assert v["detectados si se revisa el 20 %"] == pytest.approx(100)


def test_calibration_summary_flags_an_overconfident_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.9, 5000)
    honest = pd.DataFrame({"y_true": (rng.random(5000) < p).astype(int), "proba": p})                 # dice lo que pasa
    overconfident = honest.assign(proba=np.clip(p + 0.15, 0, 1))                                      # sube todo 15 puntos
    summary = calibration_summary({"honesto": honest, "sobreconfiado": overconfident}, ["honesto", "sobreconfiado"])
    assert summary.loc["honesto", "error de calibración"] < 0.03 < summary.loc["sobreconfiado", "error de calibración"]
    assert summary.loc["honesto", "predicciones cerca del umbral"] > 0


def test_calibration_by_bin_uses_spanish_labels():
    preds = {"m": pd.DataFrame({"y_true": [0, 0, 1, 1] * 5, "proba": np.linspace(0.05, 0.95, 20)})}
    assert list(calibration_by_bin(preds, "m", n_bins=4).columns) == ["predicciones", "prob. predicha", "tasa observada"]


def test_is_baseline():
    assert is_baseline("línea base: prior") and not is_baseline("LightGBM")


def test_technical_table_also_blanks_the_pay_delay_rule_and_every_rule_name_exists_as_a_candidate():
    from proyecto.models.candidates import tournament_candidates
    from proyecto.models.tournament import NO_PROBABILITY
    per_fold = {"línea base: regla PAY_0 ≥ 2": _fold_frame([0.64, 0.65]), "LightGBM": _fold_frame([0.78, 0.79])}
    table = technical_table(per_fold, order=list(per_fold))
    assert table.loc["línea base: regla PAY_0 ≥ 2", ["log loss", "Brier", "prob. media predicha"]].isna().all()
    assert not table.loc["LightGBM", ["log loss", "Brier", "prob. media predicha"]].isna().any()
    candidates = tournament_candidates()
    assert set(NO_PROBABILITY) <= set(candidates)                      # si se renombra una línea base, esta prueba avisa
    assert all(is_baseline(name) for name in NO_PROBABILITY)


def test_calibration_summary_with_hand_computed_values():
    p = [0.10, 0.15, 0.20, 0.30, 0.50, 0.90]
    y = [0, 0, 1, 0, 1, 1]
    row = calibration_summary({"m": pd.DataFrame({"y_true": y, "proba": p})}, ["m"], n_bins=2).loc["m"]
    assert row["predicciones cerca del umbral"] == 2                      # 0,15 y 0,20 están en [0,12; 0,22)
    assert row["prob. predicha ahí"] == pytest.approx(0.175) and row["tasa observada ahí"] == pytest.approx(0.5)
    # 2 tramos de 3: (0,10; 0,15; 0,20) predicho 0,15 y observado 1/3 | (0,30; 0,50; 0,90) predicho 0,5667 y observado 2/3
    assert row["error de calibración"] == pytest.approx((abs(0.15 - 1 / 3) + abs(17 / 30 - 2 / 3)) / 2)


def test_variant_labels_match_the_real_number_of_variables():
    from proyecto.data.cleaning import PREDICTORS
    from proyecto.features.build import PACKAGE_D, PACKAGE_E
    from proyecto.features.variants import REFERENCE_NAME, VARIANTS
    assert f"({len(PREDICTORS)} variables)" in REFERENCE_NAME
    label_e = [k for k, v in VARIANTS.items() if v[0] == PACKAGE_E][0]
    label_d = [k for k, v in VARIANTS.items() if v[0] == PACKAGE_D][0]
    assert f"({len(PACKAGE_E)} variables)" in label_e and f"({len(PACKAGE_D)} variables)" in label_d
