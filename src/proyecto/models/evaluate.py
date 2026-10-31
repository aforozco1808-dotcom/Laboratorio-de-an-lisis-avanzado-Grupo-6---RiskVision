"""Medición de ideas con los mismos pliegues: AUC, AUC-PR y penalización relativa, y la regla de decisión.

La penalización relativa usa la matriz [[0, 1], [5, 0]] del caso: el 5 a 1 es una HIPÓTESIS del grupo, no un
costo observado del banco (reports/00_caso.md). Se mide con un umbral fijo de antemano (1/6, el punto de
indiferencia teórico para probabilidades calibradas), no elegido mirando el pliegue de validación.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
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
