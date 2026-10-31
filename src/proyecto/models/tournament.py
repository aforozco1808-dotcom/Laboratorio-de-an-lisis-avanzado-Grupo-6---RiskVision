"""Tablas del torneo de modelos (notebook 05) a partir de los resultados por pliegue.

Las etiquetas de columnas están en español porque las tablas son para leer; los nombres del código van en inglés.
"""
import numpy as np
import pandas as pd

from proyecto.models.evaluate import calibration_table, expected_calibration_error, observed_rate_between

TECHNICAL_LABELS = {
    "roc_auc": "AUC", "avg_precision": "AUC-PR", "balanced_accuracy": "exactitud balanceada", "f1_macro": "F1 macro",
    "precision_1": "precisión (incumple)", "recall_0": "recall (paga)", "recall_1": "recall (incumple)",
    "flagged_share": "% marcado", "penalty_2": "penalización 2:1", "penalty": "penalización 5:1",
    "penalty_10": "penalización 10:1", "log_loss": "log loss", "brier": "Brier",
    "mean_proba": "prob. media predicha", "recall_top_5": "recall al revisar 5 %", "recall_top_10": "recall al revisar 10 %",
    "recall_top": "recall al revisar 20 %", "lift_top": "lift al revisar 20 %",
}
LOWER_IS_BETTER = {"penalización 2:1", "penalización 5:1", "penalización 10:1", "log loss", "Brier"}
NO_PROBABILITY = ("línea base: siempre paga", "línea base: regla PAY_0 ≥ 2")   # dan 0 o 1: sin log loss ni Brier
THRESHOLD_ZONE = (0.12, 0.22)   # «zona del umbral»: alrededor de 1/6 ≈ 0,167. Si cambia el 5 a 1, hay que revisarla a mano


def is_baseline(name: str) -> bool:
    return name.startswith("línea base")


def summarize_by_model(per_fold: dict, metric: str = "roc_auc") -> pd.DataFrame:
    """Media, variación entre pliegues y segundos por ajuste de cada modelo, ordenados por la métrica (mayor primero)."""
    table = pd.DataFrame({
        name: {"ROC AUC (media)": f[metric].mean(), "± entre pliegues": f[metric].std(ddof=1),
               "segundos por ajuste": f["fit_seconds"].mean()}
        for name, f in per_fold.items()
    }).T
    return table.sort_values("ROC AUC (media)", ascending=False)


def technical_table(per_fold: dict, order, labels: dict = TECHNICAL_LABELS, no_probability=NO_PROBABILITY) -> pd.DataFrame:
    """Media de las métricas técnicas de cada modelo. Para las reglas que dan 0 o 1 se vacían log loss, Brier y la
    probabilidad media, que no tienen sentido."""
    table = pd.DataFrame({n: f[list(labels)].mean() for n, f in per_fold.items()}).T.rename(columns=labels).loc[list(order)]
    rows = [n for n in table.index if n in no_probability]
    table.loc[rows, ["log loss", "Brier", "prob. media predicha"]] = np.nan
    return table


def rank_positions(table: pd.DataFrame, columns, lower_is_better=LOWER_IS_BETTER) -> pd.DataFrame:
    """Posición de cada modelo (1 = mejor) en cada métrica. Los valores vacíos quedan vacíos."""
    return pd.DataFrame({
        col: table[col].rank(ascending=col in lower_is_better, method="min").astype("Int64") for col in columns
    })


def best_by_scenario(table: pd.DataFrame, columns, models, lower_is_better=LOWER_IS_BETTER) -> dict:
    """Para cada columna, el mejor de `models` y su valor: {columna: (modelo, valor)}."""
    out = {}
    for col in columns:
        values = table.loc[list(models), col]
        name = values.idxmin() if col in lower_is_better else values.idxmax()
        out[col] = (name, float(values[name]))
    return out


def calibration_summary(predictions: dict, models, low: float = THRESHOLD_ZONE[0], high: float = THRESHOLD_ZONE[1],
                        n_bins: int = 10) -> pd.DataFrame:
    """Error de calibración de cada modelo y qué pasa en la zona del umbral (probabilidad entre `low` y `high`).

    predictions: modelo -> tabla con y_true y proba de los pliegues juntos (un cliente puede aparecer varias veces,
    una por repetición, por eso se cuentan predicciones y no clientes).
    """
    rows = []
    for name in models:
        p = predictions[name]
        near = observed_rate_between(p["y_true"], p["proba"], low, high)
        rows.append({"modelo": name,
                     "error de calibración": expected_calibration_error(calibration_table(p["y_true"], p["proba"], n_bins)),
                     "predicciones cerca del umbral": near["n_predictions"],
                     "prob. predicha ahí": near["mean_predicted"], "tasa observada ahí": near["observed_rate"]})
    return pd.DataFrame(rows).set_index("modelo")


CALIBRATION_LABELS = {"n_clients": "predicciones", "mean_predicted": "prob. predicha", "observed_rate": "tasa observada"}


def calibration_by_bin(predictions: dict, model: str, n_bins: int = 10) -> pd.DataFrame:
    """Tabla por tramos de riesgo de un modelo, con etiquetas en español."""
    p = predictions[model]
    return calibration_table(p["y_true"], p["proba"], n_bins).rename(columns=CALIBRATION_LABELS)


def volume_per_thousand(table: pd.DataFrame, positive_rate: float, per: int = 1000) -> pd.DataFrame:
    """Traduce las métricas a conteos por cada `per` clientes (con el umbral fijo de la tabla técnica)."""
    positives = positive_rate * per
    flagged = table["% marcado"] * per
    detected = table["recall (incumple)"] * positives
    return pd.DataFrame({
        "se marcan para revisar": flagged,
        "incumplidores detectados": detected,
        "incumplidores que se escapan": positives - detected,
        "falsas alarmas": flagged - detected,
        "penalización total (5:1)": table["penalización 5:1"] * per,
        "detectados si se revisa el 20 %": table["recall al revisar 20 %"] * positives,
    })
