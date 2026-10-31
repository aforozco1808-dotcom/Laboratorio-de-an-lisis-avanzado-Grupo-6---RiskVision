import pytest
from sklearn.calibration import CalibratedClassifierCV

from proyecto.features.build import FINAL_FEATURES, PACKAGE_D
from proyecto.models.baselines import PayDelayRule
from proyecto.models.candidates import (
    LGBM_PARAMS,
    LOGISTIC_PARAMS,
    make_calibrated_lgbm,
    make_forest,
    make_lgbm,
    make_logistic,
    make_xgb,
    reference_models,
    tournament_candidates,
)
from tests.test_pipeline import _clients


@pytest.mark.parametrize("factory", [make_lgbm, make_logistic, make_forest, make_xgb, make_calibrated_lgbm])
def test_every_factory_fits_and_predicts_probabilities_from_raw_rows(factory):
    X, y = _clients(300)
    model = factory(FINAL_FEATURES).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_tournament_has_three_baselines_and_the_expected_candidates():
    candidates = tournament_candidates()
    baselines = [name for name in candidates if name.startswith("línea base")]
    assert len(baselines) == 3 and len(candidates) == 9
    assert {"regresión logística", "bosque aleatorio", "LightGBM", "XGBoost", "LightGBM + paquete D", "LightGBM calibrado"} <= set(candidates)
    assert isinstance(candidates["línea base: regla PAY_0 ≥ 2"](), PayDelayRule)


def test_each_call_builds_a_new_model_so_folds_never_share_a_fitted_one():
    factory = tournament_candidates()["LightGBM"]
    assert factory() is not factory()


def test_parameters_are_the_shared_ones():
    pipe = tournament_candidates()["LightGBM"]()
    assert pipe.named_steps["model"].get_params()["n_estimators"] == LGBM_PARAMS["n_estimators"]
    assert pipe.named_steps["model"].get_params()["learning_rate"] == LGBM_PARAMS["learning_rate"]
    logistic = tournament_candidates()["regresión logística"]()
    assert logistic.named_steps["model"].get_params()["C"] == LOGISTIC_PARAMS["C"]


def test_the_variant_with_package_d_adds_more_columns_than_the_final_package():
    X, y = _clients(300)
    e = tournament_candidates()["LightGBM"]().fit(X, y).named_steps["features"].transform(X)
    d = tournament_candidates()["LightGBM + paquete D"]().fit(X, y).named_steps["features"].transform(X)
    assert set(FINAL_FEATURES) <= set(e.columns) and set(PACKAGE_D) <= set(d.columns)
    assert d.shape[1] - e.shape[1] == len(PACKAGE_D) - len(FINAL_FEATURES)


def test_calibrated_candidate_is_a_calibration_wrapper_with_three_inner_folds():
    model = tournament_candidates()["LightGBM calibrado"]()
    assert isinstance(model, CalibratedClassifierCV) and model.method == "isotonic" and model.cv == 3


def test_reference_models_take_features_and_drop():
    models = reference_models()
    assert set(models) == {"LightGBM", "Logística"}
    X, y = _clients(300)
    pipe = models["Logística"](("n_delayed_months",), ("SEX",)).fit(X, y)
    columns = pipe.named_steps["features"].transform(X).columns
    assert "n_delayed_months" in columns and "SEX" not in columns


def test_every_candidate_uses_the_full_shared_parameter_set():
    """Un cambio en los parámetros de cualquier modelo movería todas las cifras sin aviso: esta prueba lo avisa."""
    from proyecto.models.candidates import RF_PARAMS, XGB_PARAMS
    for model, params in [(make_lgbm(), LGBM_PARAMS), (make_forest(), RF_PARAMS), (make_xgb(), XGB_PARAMS),
                          (make_logistic(), LOGISTIC_PARAMS)]:
        actual = model.named_steps["model"].get_params()
        assert all(actual[k] == v for k, v in params.items()), type(model.named_steps["model"]).__name__
