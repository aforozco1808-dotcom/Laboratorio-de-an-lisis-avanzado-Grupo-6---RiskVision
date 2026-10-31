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
THEORETICAL_THRESHOLD = 1 / 6   # = 1 / (1 + 5); en la igualdad se marca al cliente


def relative_penalty(y_true, proba, threshold: float = THEORETICAL_THRESHOLD, fn_weight: float = FN_WEIGHT) -> float:
    """Penalización media por cliente: (fn_weight × falsos negativos + falsos positivos) / clientes."""
    y_true, flagged = np.asarray(y_true), np.asarray(proba) >= threshold
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
COST_RATIOS = (2, 5, 10)   # escenarios de sensibilidad: cuántas veces pesa no detectar a quien incumple frente a una falsa alarma


def _penalty(y_true, proba, fn_weight: float, threshold: float | None = None) -> float:
    """Penalización media por cliente con umbral `threshold` (por defecto el teórico 1 / (1 + fn_weight))."""
    threshold = 1 / (1 + fn_weight) if threshold is None else threshold
    flagged = proba >= threshold
    fn = int((~flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return (fn_weight * fn + fp) / len(y_true)


def _top_share(y_true, proba, share: float) -> tuple[float, float]:
    """Recall y lift al revisar el `share` de clientes con más probabilidad. Los empates se desempatan por posición."""
    n_top = max(1, int(round(share * len(y_true))))
    top = np.argsort(-proba, kind="stable")[:n_top]
    recall = float(y_true[top].sum() / y_true.sum()) if y_true.sum() else 0.0
    lift = float(y_true[top].mean() / y_true.mean()) if y_true.mean() else 0.0
    return recall, lift


def technical_metrics(y_true, proba, threshold: float = THEORETICAL_THRESHOLD, review_share: float = REVIEW_SHARE) -> dict:
    """Métricas técnicas de un modelo binario en un conjunto de validación.

    - Con probabilidades (no dependen de decidir): ROC AUC, AUC-PR, log loss, Brier y tasa predicha media.
    - Con la decisión «marcar si proba ≥ threshold»: exactitud balanceada, F1 macro, precisión y recall por clase,
      porcentaje marcado y penalización relativa. El umbral 1/6 solo es el punto óptimo si las probabilidades están
      calibradas, así que favorece a los modelos mejor calibrados.
    - Sensibilidad al costo: penalización con 2 a 1 y 10 a 1 (cada una con su umbral teórico 1/3 y 1/11).
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
        "penalty": (FN_WEIGHT * fn + fp) / len(y_true),
        "penalty_2": _penalty(y_true, proba, 2),
        "penalty_10": _penalty(y_true, proba, 10),
        "recall_top_5": _top_share(y_true, proba, 0.05)[0],
        "recall_top_10": _top_share(y_true, proba, 0.10)[0],
        "recall_top": recall_top,
        "lift_top": lift_top,
    }


def score_folds_detailed(make_pipeline, X, y, folds, keep_predictions: bool = False):
    """Como `score_folds`, pero con todas las métricas técnicas y los segundos de ajuste de cada pliegue.

    Con `keep_predictions=True` devuelve también una tabla (pliegue, fila, y_true, proba) con las predicciones de
    validación de cada pliegue, para revisar la calibración sin volver a ajustar nada.
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
    """Calibración por tramos: agrupa los clientes en `n_bins` grupos del mismo tamaño según su probabilidad
    predicha y compara la probabilidad media predicha con la tasa que de verdad incumplió en cada grupo."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba, dtype=float)})
    frame["tramo"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    table = frame.groupby("tramo").agg(clientes=("y", "size"), prob_predicha=("p", "mean"), tasa_observada=("y", "mean"))
    return table.reset_index(drop=True)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Diferencia absoluta media entre lo predicho y lo observado por tramo, ponderada por clientes (0 = perfecta)."""
    gap = (table["prob_predicha"] - table["tasa_observada"]).abs()
    return float((gap * table["clientes"]).sum() / table["clientes"].sum())


def observed_rate_between(y_true, proba, low: float, high: float) -> dict:
    """Entre los clientes con probabilidad predicha en [low, high): cuántos son, qué probabilidad media se les dio
    y qué tasa de incumplimiento tuvieron. Sirve para ver si la zona del umbral está bien calibrada."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    inside = (proba >= low) & (proba < high)
    n = int(inside.sum())
    return {"clientes": n, "prob_predicha": float(proba[inside].mean()) if n else float("nan"),
            "tasa_observada": float(y_true[inside].mean()) if n else float("nan")}
