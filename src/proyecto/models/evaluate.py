"""Medición de ideas con los mismos pliegues: AUC, AUC-PR y penalización relativa, y la regla de decisión.

La penalización relativa usa la matriz [[0, 1], [5, 0]] del caso: el 5 a 1 es una HIPÓTESIS del grupo, no un
costo observado del banco (reports/00_caso.md). Se mide con un umbral fijo de antemano (1/6, el punto de
indiferencia teórico para probabilidades calibradas), no elegido mirando el pliegue de validación.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from proyecto.config import RANDOM_STATE

FN_WEIGHT = 5            # hipótesis del grupo: no detectar a quien incumple pesa 5 veces una falsa alarma


def theoretical_threshold(fn_weight: float) -> float:
    """Umbral de indiferencia de la matriz [[0, 1], [fn_weight, 0]]: 1 / (1 + fn_weight). Solo vale con probabilidades calibradas."""
    return 1 / (1 + fn_weight)


THEORETICAL_THRESHOLD = theoretical_threshold(FN_WEIGHT)   # 1/6 con 5 a 1; en la igualdad se marca al cliente


def relative_penalty(y_true, proba, threshold: float = THEORETICAL_THRESHOLD, fn_weight: float = FN_WEIGHT) -> float:
    """Penalización media por cliente: (fn_weight × falsos negativos + falsos positivos) / clientes.

    Es la ÚNICA cuenta de penalización del proyecto: se marca al cliente si proba ≥ threshold.
    """
    y_true, flagged = np.asarray(y_true), np.asarray(proba, dtype=float) >= threshold
    fn = int(((~flagged) & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return (fn_weight * fn + fp) / len(y_true)


def repeated_folds(X, y, n_splits: int = 5, n_repeats: int = 3, seed: int = RANDOM_STATE):
    """Pliegues estratificados repetidos. Se calculan una vez y se reutilizan en toda comparación."""
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X, y))


def score_folds(make_pipeline, X, y, folds) -> pd.DataFrame:
    """Ajusta un Pipeline nuevo en cada pliegue (solo con su entrenamiento) y mide en su validación."""
    y = np.asarray(y)
    rows = []
    for train_idx, valid_idx in folds:
        pipe = make_pipeline().fit(X.iloc[train_idx], y[train_idx])
        proba = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        rows.append({"roc_auc": roc_auc_score(y[valid_idx], proba),
                     "avg_precision": average_precision_score(y[valid_idx], proba),
                     "penalty": relative_penalty(y[valid_idx], proba)})
    return pd.DataFrame(rows)


def improvement_over_noise(reference, variant, higher_is_better: bool = True) -> dict:
    """Compara una variante con la referencia sobre los mismos pliegues.

    Devuelve la mejora media (positiva si mejora) y dos cocientes:
      - gain_over_noise: mejora / desviación del AUC de la variante entre pliegues (regla original del curso).
        El «ruido» incluye lo difícil que es cada pliegue, que afecta igual a la referencia y a la variante.
      - paired_gain: mejora / desviación de las DIFERENCIAS pliegue a pliegue (regla vigente desde 2026-10-31).
        Compara cada pliegue consigo mismo, así que no arrastra esa dificultad compartida.
    También cuenta en cuántos pliegues la variante supera a la referencia.
    """
    reference, variant = np.asarray(reference, dtype=float), np.asarray(variant, dtype=float)
    diff = (1 if higher_is_better else -1) * (variant - reference)
    mean_gain = float(diff.mean())

    def ratio(den):
        return mean_gain / den if den > 0 else (np.inf if mean_gain > 0 else 0.0)

    noise = float(np.std(variant, ddof=1))
    return {"gain": mean_gain, "noise": noise, "gain_over_noise": ratio(noise),
            "paired_gain": ratio(float(np.std(diff, ddof=1))),
            "folds_improved": int((diff > 0).sum()), "n_folds": len(diff)}


def decide(per_model: dict, min_ratio: float = 1.0, min_share: float = 0.8, key: str = "paired_gain"):
    """Regla de la especificación (§3): se conserva si en AL MENOS UN modelo el cociente `key` es > min_ratio
    y la variante mejora en ≥ min_share de los pliegues, y en NINGÚN modelo empeora en promedio.

    key="paired_gain" es la regla vigente; key="gain_over_noise" es la original y se conserva para comparar.
    per_model: {"modelo": salida de improvement_over_noise}. Devuelve (conservar, motivo).
    """
    worse = [m for m, r in per_model.items() if r["gain"] < 0]
    if worse:
        return False, f"empeora en {', '.join(worse)}"
    passing = [m for m, r in per_model.items()
               if r[key] > min_ratio and r["folds_improved"] >= min_share * r["n_folds"]]
    if passing:
        return True, f"cumple la regla en {', '.join(passing)} y no empeora en los demás"
    return False, "no supera el ruido en ningún modelo"


REVIEW_SHARE = 0.20   # capacidad supuesta de revisión: el 20 % de clientes con más riesgo (escenario del EDA, no una medición)


def _top_share(y_true, proba, share: float) -> tuple[float, float]:
    """Recall y lift al revisar el `share` de clientes con más probabilidad.

    Los empates en el corte se tratan por valor esperado: si varios clientes tienen exactamente la misma probabilidad y
    solo caben algunos, cada uno aporta la fracción que le corresponde. Así el resultado no depende del orden de las
    filas (una línea base constante da recall = share y lift = 1).
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    n = len(y_true)
    n_top = min(n, max(1, int(round(share * n))))
    cutoff = proba[np.argsort(-proba, kind="stable")[n_top - 1]]
    above, tied = proba > cutoff, proba == cutoff
    slots = n_top - int(above.sum())                        # lugares que quedan para el grupo empatado (siempre ≥ 1)
    expected_hits = y_true[above].sum() + y_true[tied].sum() * slots / int(tied.sum())
    recall = float(expected_hits / y_true.sum()) if y_true.sum() else 0.0
    lift = float((expected_hits / n_top) / y_true.mean()) if y_true.mean() else 0.0
    return recall, lift


def technical_metrics(y_true, proba, threshold: float = THEORETICAL_THRESHOLD, review_share: float = REVIEW_SHARE) -> dict:
    """Métricas técnicas de un modelo binario en un conjunto de validación.

    - Con probabilidades (no dependen de decidir): ROC AUC, AUC-PR, log loss, Brier y tasa predicha media.
    - Con la decisión «marcar si proba ≥ threshold»: exactitud balanceada, F1 macro, precisión y recall por clase,
      porcentaje marcado y la penalización 5 a 1 (`penalty`). El umbral 1/6 solo es el punto óptimo si las
      probabilidades están calibradas, así que favorece a los modelos mejor calibrados.
    - Sensibilidad al costo: `penalty_2` y `penalty_10` usan SIEMPRE su propio umbral teórico (1/3 y 1/11), sin
      importar `threshold`.
    - Con capacidad fija (revisar el 5 %, 10 % o `review_share` de mayor riesgo): recall y lift. Comparan solo el
      orden, sin depender de la calibración.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    flagged = proba >= threshold
    tp, fp = int((flagged & (y_true == 1)).sum()), int((flagged & (y_true == 0)).sum())
    fn, tn = int((~flagged & (y_true == 1)).sum()), int((~flagged & (y_true == 0)).sum())
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 else 0.0
    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if precision_0 + recall_0 else 0.0

    recall_top, lift_top = _top_share(y_true, proba, review_share)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
        "mean_proba": float(proba.mean()),
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "f1_macro": (f1_0 + f1_1) / 2,
        "precision_1": precision_1,
        "recall_0": recall_0,
        "recall_1": recall_1,
        "flagged_share": float(flagged.mean()),
        "penalty": relative_penalty(y_true, proba, threshold, FN_WEIGHT),
        "penalty_2": relative_penalty(y_true, proba, theoretical_threshold(2), 2),
        "penalty_10": relative_penalty(y_true, proba, theoretical_threshold(10), 10),
        "recall_top_5": _top_share(y_true, proba, 0.05)[0],
        "recall_top_10": _top_share(y_true, proba, 0.10)[0],
        "recall_top": recall_top,
        "lift_top": lift_top,
    }


def score_folds_detailed(make_pipeline, X, y, folds, keep_predictions: bool = False):
    """Como `score_folds`, pero con todas las métricas técnicas y los segundos de ajuste de cada pliegue.

    Con `keep_predictions=True` devuelve también una tabla (fold, row, y_true, proba) con las predicciones de
    validación de cada pliegue, para revisar la calibración sin volver a ajustar nada. `row` es la posición de la fila
    en `X`, no el ID del cliente.
    """
    y = np.asarray(y)
    rows, preds = [], []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        pipe = make_pipeline()
        start = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        seconds = time.perf_counter() - start
        proba = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        rows.append({**technical_metrics(y[valid_idx], proba), "fit_seconds": seconds})
        if keep_predictions:
            preds.append(pd.DataFrame({"fold": fold, "row": valid_idx, "y_true": y[valid_idx], "proba": proba}))
    scores = pd.DataFrame(rows)
    return (scores, pd.concat(preds, ignore_index=True)) if keep_predictions else scores


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Calibración por tramos: agrupa a los clientes en hasta `n_bins` grupos según su probabilidad predicha y compara la
    probabilidad media predicha (`mean_predicted`) con la tasa que de verdad incumplió (`observed_rate`) en cada grupo.

    Los tramos se arman por VALOR de probabilidad: clientes con la misma probabilidad quedan siempre juntos. Cada valor
    distinto va al tramo donde cae su posición central en el orden de menor a mayor. Con probabilidades todas distintas
    los tramos salen del mismo tamaño; con muchos empates (por ejemplo, una calibración isotónica) puede haber menos de
    `n_bins` tramos y de distinto tamaño, pero nunca se funden dos valores distintos que deberían quedar separados.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    n = len(y_true)
    if n < n_bins:
        raise ValueError(f"Hay menos clientes ({n}) que tramos ({n_bins})")
    _, inverse, counts = np.unique(proba, return_inverse=True, return_counts=True)
    centers = (np.cumsum(counts) - counts / 2) / n                       # posición relativa (0 a 1) del centro de cada valor
    bin_of_value = np.minimum((centers * n_bins).astype(int), n_bins - 1)
    frame = pd.DataFrame({"y": y_true, "p": proba, "bin": bin_of_value[inverse]})
    table = frame.groupby("bin").agg(n_clients=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return table.reset_index(drop=True)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Diferencia absoluta media entre lo predicho y lo observado por tramo, ponderada por clientes (0 = perfecta)."""
    gap = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float((gap * table["n_clients"]).sum() / table["n_clients"].sum())


def observed_rate_between(y_true, proba, low: float, high: float) -> dict:
    """Entre las predicciones con probabilidad en [low, high) (`low` incluido, `high` excluido): cuántas son, qué
    probabilidad media se les dio y qué tasa de incumplimiento tuvieron. Sirve para ver si la zona del umbral está bien
    calibrada. Con las predicciones de varias repeticiones juntas, un mismo cliente puede contarse varias veces."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    inside = (proba >= low) & (proba < high)
    n = int(inside.sum())
    return {"n_predictions": n, "mean_predicted": float(proba[inside].mean()) if n else float("nan"),
            "observed_rate": float(y_true[inside].mean()) if n else float("nan")}
