# Plan · 05 Torneo de modelos

**Entrada:** `especificaciones/05-torneo.md`   **Estado:** ☒ borrador (pendiente de revisión del grupo)

## Resumen
Medir 5 modelos y 2 líneas base con los mismos 15 pliegues y las mismas variables, guardar cada pliegue, comparar de a pares y elegir 1–2 finalistas para la etapa 6. Es una etapa de medición: no se optimiza nada.

## Decisiones de diseño
| Decisión | Motivo |
|---|---|
| Reutilizar `repeated_folds`, `build_pipeline` y `improvement_over_noise` de la etapa 4 | Misma validación y misma medida de diferencia, así los resultados son comparables |
| ROC AUC decide; la penalización 5:1 solo controla | La ficha del caso fija el AUC; el 5:1 es una hipótesis |
| Recall/lift al revisar el 20 % como control extra | El umbral 1/6 depende de la calibración; esta medida compara solo el orden |
| Volumen por cada 1.000 clientes, no mensual | El grupo no tiene un dato del volumen del banco; inventarlo contradice la regla 1 de `CLAUDE.md` |
| Variante LightGBM + paquete D | Cierra el pendiente de la etapa 4 sin cambiar las variables del resto |
| Resultado por pliegue en un CSV versionado | Trazabilidad: cualquiera puede revisar las cifras sin volver a correr 10 minutos |

## Estructura que se agrega
```text
src/proyecto/models/evaluate.py   # + technical_metrics, score_folds_detailed, calibration_table y afines
src/proyecto/models/baselines.py  # PayDelayRule (regla PAY_0 >= 2)
tests/test_evaluate.py            # + pruebas con valores calculados a mano
notebooks/05_torneo.ipynb         # orquesta el torneo
reports/05_torneo.md              # informe
reports/05_torneo_por_pliegue.csv # una fila por modelo y pliegue
```

## Riesgos
| Riesgo | Mitigación |
|---|---|
| Parámetros de partida que favorecen a un modelo | Esfuerzo parecido para todos y aviso de que no están afinados |
| El umbral 1/6 premia la calibración, no el orden | Se agrega recall al revisar el 20 % y se muestran log loss, Brier y la tasa predicha |
| Elegir por diferencias dentro del ruido | Regla de finalistas escrita antes, con diferencia pareada y pliegues ganados |
| Los 15 pliegues comparten filas | La diferencia pareada puede salir algo inflada; se dice en el informe |
| La prueba ya está expuesta | No se abre; el resultado final se reportará como complementario |
