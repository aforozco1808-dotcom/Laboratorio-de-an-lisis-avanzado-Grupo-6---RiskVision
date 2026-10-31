"""Comparación de variantes y de modelos sobre los mismos pliegues (notebooks 04 y 05).

Las funciones devuelven tablas con etiquetas de columnas en español porque son para leer; los nombres del código van en inglés.
"""
import pandas as pd

from proyecto.models.evaluate import decide, improvement_over_noise, repeated_folds, score_folds


def measure_variants(models: dict, variants: dict, X, y, folds) -> dict:
    """Mide cada variante en cada modelo con los mismos pliegues.

    models: nombre -> función (features, drop) que arma el Pipeline. variants: nombre -> (features, drop).
    Devuelve {modelo: {variante: DataFrame con una fila por pliegue}}.
    """
    results = {name: {} for name in models}
    for model_name, factory in models.items():
        for variant_name, (features, drop) in variants.items():
            results[model_name][variant_name] = score_folds(lambda f=features, d=drop, fab=factory: fab(f, d), X, y, folds)
    return results


def improvement_table(results: dict, variants: dict, models: dict, reference: str, metric: str = "roc_auc"):
    """Tabla de mejora de cada variante frente a la referencia, por modelo, con la decisión según las dos reglas.

    Devuelve (tabla, comparaciones), donde comparaciones[variante][modelo] es la salida de `improvement_over_noise`.
    """
    rows, comparisons = [], {}
    for name in variants:
        row = {"variante": name}
        comparisons[name] = {}
        for model in models:
            ref, var = results[model][reference][metric], results[model][name][metric]
            r = improvement_over_noise(ref, var)
            comparisons[name][model] = r
            row[f"{model}: AUC"] = var.mean()
            row[f"{model}: mejora"] = r["gain"]
            row[f"{model}: pareada"] = r["paired_gain"]
            row[f"{model}: mejora/ruido"] = r["gain_over_noise"]
            row[f"{model}: pliegues"] = f"{r['folds_improved']}/{r['n_folds']}"
        if name != reference:
            row["decisión (pareada)"] = "conservar" if decide(comparisons[name])[0] else "descartar"
            row["decisión (original)"] = "conservar" if decide(comparisons[name], key="gain_over_noise")[0] else "descartar"
        else:
            row["decisión (pareada)"] = row["decisión (original)"] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("variante"), comparisons


def control_table(results: dict, variants: dict, models: dict, reference: str) -> pd.DataFrame:
    """AUC-PR y penalización 5:1 de cada variante y su diferencia con la referencia (positivo = mejor)."""
    rows = []
    for model in models:
        ref = results[model][reference]
        for name in variants:
            d = results[model][name]
            rows.append({"modelo": model, "variante": name,
                         "AUC-PR": d["avg_precision"].mean(),
                         "Δ AUC-PR": (d["avg_precision"] - ref["avg_precision"]).mean(),
                         "penalización": d["penalty"].mean(),
                         "Δ penalización (positivo = mejor)": (ref["penalty"] - d["penalty"]).mean()})
    return pd.DataFrame(rows).set_index(["modelo", "variante"])


def confirm_on_new_folds(X, y, models: dict, features, seeds=(7, 99), metric: str = "roc_auc") -> dict:
    """Repite la comparación «referencia contra paquete de variables» con pliegues de otras semillas.

    Devuelve {semilla: (resultado por modelo, conserva con la regla pareada, conserva con la regla original)}.
    Los clientes son los mismos, así que reduce el sesgo de selección pero no lo elimina.
    """
    out = {}
    for seed in seeds:
        new_folds = repeated_folds(X, y, n_splits=5, n_repeats=3, seed=seed)
        per_model = {}
        for name, factory in models.items():
            ref = score_folds(lambda fab=factory: fab((), ()), X, y, new_folds)[metric]
            var = score_folds(lambda fab=factory: fab(features, ()), X, y, new_folds)[metric]
            per_model[name] = improvement_over_noise(ref, var)
        out[seed] = (per_model, decide(per_model)[0], decide(per_model, key="gain_over_noise")[0])
    return out


def compare_pair_on_new_folds(X, y, make_reference, make_variant, seeds=(7, 99), metric: str = "roc_auc") -> dict:
    """Variante frente a referencia (dos funciones sin argumentos que arman un modelo) con pliegues de otras semillas."""
    out = {}
    for seed in seeds:
        new_folds = repeated_folds(X, y, n_splits=5, n_repeats=3, seed=seed)
        ref = score_folds(make_reference, X, y, new_folds)[metric]
        var = score_folds(make_variant, X, y, new_folds)[metric]
        out[seed] = improvement_over_noise(reference=ref, variant=var)
    return out


def pairwise_table(per_fold: dict, pairs, metric: str = "roc_auc") -> pd.DataFrame:
    """Compara de a pares modelos ya medidos en los mismos pliegues. Cada par se ordena para que aparezca primero el de
    mayor media; la diferencia pareada es diferencia media ÷ variación de esa diferencia entre pliegues."""
    rows = []
    for a, b in pairs:
        best, other = (a, b) if per_fold[a][metric].mean() >= per_fold[b][metric].mean() else (b, a)
        r = improvement_over_noise(reference=per_fold[other][metric], variant=per_fold[best][metric])
        rows.append({"mejor": best, "contra": other, "diferencia media de AUC": r["gain"],
                     "pliegues que gana": f"{r['folds_improved']} de {r['n_folds']}", "diferencia pareada": r["paired_gain"]})
    return pd.DataFrame(rows)
