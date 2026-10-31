"""Modelos candidatos y sus parámetros fijos de partida.

Un solo lugar para los parámetros: los notebooks 04 (variables) y 05 (torneo) los importan de aquí, así que no pueden
quedar distintos entre sí. Son valores de partida con esfuerzo parecido para todos, NO están afinados (eso es la etapa 6).
"""
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from proyecto.config import RANDOM_STATE
from proyecto.features.build import FINAL_FEATURES, PACKAGE_D
from proyecto.models.baselines import PayDelayRule
from proyecto.models.train import build_pipeline

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1, n_jobs=4)
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=RANDOM_STATE,
                  n_jobs=4, verbosity=0)
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=30, max_features="sqrt", n_jobs=4, random_state=RANDOM_STATE)
LOGISTIC_PARAMS = dict(C=0.5, max_iter=2000)


def make_lgbm(features=(), drop=()):
    """LightGBM dentro del Pipeline común."""
    return build_pipeline(LGBMClassifier(**LGBM_PARAMS), features, drop)


def make_logistic(features=(), drop=()):
    """Regresión logística: números estandarizados y `PAY_*` como estados (no como cantidades)."""
    return build_pipeline(LogisticRegression(**LOGISTIC_PARAMS), features, drop, linear=True, pay_as_category=True)


def make_forest(features=()):
    """Bosque aleatorio dentro del Pipeline común."""
    return build_pipeline(RandomForestClassifier(**RF_PARAMS), features)


def make_xgb(features=()):
    """XGBoost dentro del Pipeline común."""
    return build_pipeline(XGBClassifier(**XGB_PARAMS), features)


def make_calibrated_lgbm(features=()):
    """LightGBM con calibración isotónica ajustada solo con el entrenamiento de cada pliegue (3 pliegues internos).

    OJO: con `ensemble=True` (valor por defecto) el modelo final es el promedio de 3 LightGBM entrenados con 2/3 del
    entrenamiento cada uno. Su diferencia con el LightGBM normal mezcla dos efectos: calibrar y promediar.
    """
    return CalibratedClassifierCV(make_lgbm(features), method="isotonic", cv=3)


def reference_models() -> dict:
    """Los dos modelos de referencia de la etapa 4: nombre -> función (features, drop) que arma el Pipeline."""
    return {"LightGBM": make_lgbm, "Logística": make_logistic}


def tournament_candidates() -> dict:
    """Candidatos de la etapa 5: nombre -> función sin argumentos que arma un modelo nuevo (el paquete final de
    variables para todos, salvo la variante declarada con el paquete D)."""
    return {
        "línea base: siempre paga": lambda: build_pipeline(DummyClassifier(strategy="most_frequent")),
        "línea base: prior": lambda: build_pipeline(DummyClassifier(strategy="prior")),
        "línea base: regla PAY_0 ≥ 2": lambda: PayDelayRule(min_months=2),
        "regresión logística": lambda: make_logistic(FINAL_FEATURES),
        "bosque aleatorio": lambda: make_forest(FINAL_FEATURES),
        "LightGBM": lambda: make_lgbm(FINAL_FEATURES),
        "XGBoost": lambda: make_xgb(FINAL_FEATURES),
        "LightGBM + paquete D": lambda: make_lgbm(PACKAGE_D),
        "LightGBM calibrado": lambda: make_calibrated_lgbm(FINAL_FEATURES),
    }
