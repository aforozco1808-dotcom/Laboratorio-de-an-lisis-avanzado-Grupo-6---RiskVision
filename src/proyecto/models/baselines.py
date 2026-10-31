"""Líneas base para el torneo: reglas sin aprendizaje que cualquier modelo serio debe superar."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PayDelayRule(BaseEstimator, ClassifierMixin):
    """Regla del EDA: marca como «incumple» a quien tiene `PAY_0 >= min_months` (atraso en el último mes).

    No aprende nada de los datos. Da probabilidad 1 a quien cumple la regla y 0 al resto, por lo que no son
    probabilidades reales: su log loss no tiene sentido y sus empates se desempatan por posición.
    Recibe la tabla original (con `PAY_0`), por eso no pasa por `build_pipeline`.
    """

    def __init__(self, min_months: int = 2):
        self.min_months = min_months

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        flagged = (np.asarray(X["PAY_0"], dtype=float) >= self.min_months).astype(float)
        return np.column_stack([1 - flagged, flagged])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
