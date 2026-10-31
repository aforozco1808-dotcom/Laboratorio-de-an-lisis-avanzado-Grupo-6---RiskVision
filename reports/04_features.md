# Informe de variables (etapa 4)

> Resultado de `notebooks/04_features.ipynb` (RiskVision, Grupo 6). Especificación: `especificaciones/04-features.md`.
> Código: `src/proyecto/features/build.py`, `src/proyecto/models/train.py` y `src/proyecto/models/evaluate.py`, con pruebas en `tests/`.

**Alcance.** Todo se midió con el 80 % de entrenamiento (24.000 clientes), con 5 pliegues estratificados repetidos 3 veces (15 mediciones por variante, semilla 42) y dos modelos con parámetros fijos: LightGBM y regresión logística. La prueba original no se abrió. Recordatorio de la revisión H1: esa prueba ya se había explorado y evaluado antes, por lo que no sirve como evaluación final independiente. La penalización relativa usa la hipótesis 5 a 1 del grupo y un umbral fijo de 1/6; no es un costo observado ni dinero.

## Nombres de las variables en el código
El código usa nombres en inglés (regla del proyecto). Equivalencias con los nombres que aparecen en conversaciones y borradores anteriores:

| Nombre en el código | Nombre anterior | Qué es |
|---|---|---|
| `utilization_last` | `util_1` | Factura del último mes ÷ límite |
| `utilization_mean` | `util_media` | Factura media de 6 meses ÷ límite |
| `utilization_max` | `util_max` | Factura máxima de 6 meses ÷ límite |
| `pending_balance_last` | `saldo_pend_1` | (Factura de agosto − pago de septiembre) ÷ límite |
| `delay_streak` | `racha_atraso` | Meses seguidos con atraso desde el más reciente |
| `delay_severity` | `gravedad_atraso` | Suma de los meses de atraso |
| `severe_delay_months` | `atraso_grave` | Meses con 2 o más meses de atraso |
| `n_delayed_months` | `n_atraso` | Meses con algún atraso |
| `n_zero_payments` | `n_pago_cero` | Meses sin ningún pago |
| `n_minimum_payments` | `paga_minimo` | Meses con pago ≤ 5 % de una factura positiva |
| `log_limit` | `log_limite` | Logaritmo del límite |

(`n_atraso` sigue llamándose así dentro del EDA, notebook 02, donde nació.)

## Cómo se midió
- **Modelos (parámetros fijos):** LightGBM (`n_estimators=300`, `learning_rate=0.03`, `num_leaves=15`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`) y regresión logística (`C=0.5`, `PAY_*` como estados y números estandarizados).
- **Métrica principal:** ROC AUC, porque ordena a los clientes por riesgo sin fijar un umbral (ver `reports/00_caso.md`).
- **Métricas de control:** AUC-PR y penalización relativa con la hipótesis 5 a 1 y umbral fijo de 1/6.
- **Pliegues:** 5 estratificados × 3 repeticiones (semilla 42), partidos por fila porque los 30.000 `ID` son únicos. Los mismos para todas las variantes. La repetición con otros pliegues usa las semillas 7 y 99.
- **Regla de decisión:** la de abajo. Se escribió antes de medir, pero se cambió una vez después de ver un resultado (ver «Cambio de regla»).

## Resultado en una frase
Un paquete de cinco variables (`utilization_mean`, `pending_balance_last`, `delay_streak`, `n_zero_payments`, `log_limit`) sube el AUC de la regresión logística de 0,7697 a 0,7785 y el de LightGBM de 0,7853 a 0,7870. La mejora es clara en la logística y pequeña en LightGBM. **Es una cifra optimista:** el paquete se eligió y se midió con los mismos 24.000 clientes, y las semillas 7 y 99 son esos mismos clientes en otro orden, no datos nuevos.

## Regla de decisión
Una idea o paquete se conserva si, en al menos un modelo, la **mejora pareada** (mejora media del AUC dividida entre la variación de las diferencias pliegue a pliegue) es mayor que 1 y gana en 12 de los 15 pliegues o más, si no empeora en promedio en ningún modelo y si se conoce al predecir.

**Cambio de regla (2026-10-31).** La primera versión usaba «mejora / ruido», donde el ruido era la variación del AUC de la variante entre pliegues. Con ella, el paquete E no se confirmó en la semilla 99 (0,81), aunque su mejora fue casi igual en las tres semillas (+0,0088, +0,0088, +0,0089) y ganó en los 45 pliegues de la logística (15 de 15 en cada semilla). La variación entre pliegues va de 0,006 a 0,011 según la semilla y afecta por igual a la referencia y a la variante, así que no sirve para comparar dos variantes sobre los mismos pliegues. El cambio se hizo **después de ver ese resultado**; por eso el notebook muestra las dos cifras.

## Tabla de decisiones
Cada celda de modelo: mejora del AUC (mejora pareada; pliegues en que gana).

| # | Idea | LightGBM | Regresión logística | Decisión | Motivo |
|---|---|---|---|---|---|
| 1 | Utilización (`utilization_last`, `utilization_mean`, `utilization_max`) | +0,0019 (0,97; 12/15) | +0,0007 (1,09; 13/15) | Conservar | Pasa en la logística y no empeora LightGBM; mejora pequeña. En el paquete E entra `utilization_mean` |
| 2 | `pending_balance_last` | +0,0011 (0,57; 9/15) | +0,0006 (0,88; 12/15) | No pasa sola | Va dentro de E porque suma en LightGBM junto con las demás |
| 3 | `delay_streak` | −0,0002 (−0,17; 6/15) | +0,0021 (2,15; 15/15) | No pasa sola | LightGBM baja 0,0002 (ruido); ayuda a la logística. Va en E |
| 4 | `n_zero_payments` | +0,0003 (0,26; 8/15) | +0,0045 (2,86; 15/15) | Conservar | La mejor idea individual para la logística |
| 5 | `log_limit` | −0,0004 (−0,31; 6/15) | +0,0030 (2,47; 15/15) | No pasa sola | Solo ayuda al modelo lineal. Va en E |
| 6 | `delay_severity`, `severe_delay_months`, `n_delayed_months` | ≈ 0 | ≈ 0 | Descartar | `PAY_*` ya contiene la información |
| 7 | `n_minimum_payments` | −0,0001 (−0,07; 8/15) | −0,0002 (−0,45; 5/15) | Descartar | No aporta; además depende de un 5 % supuesto |
| 8 | **Paquete E** | **+0,0017 (0,87; 13/15)** | **+0,0088 (5,00; 15/15)** | **Conservar** | Cumple en la logística y no empeora LightGBM |
| 9 | Paquete D (10 variables) | +0,0032 (1,67; 15/15) | +0,0084 (3,97; 15/15) | Alternativa | Pasa en ambos modelos; no se repitió con otros pliegues (no se midió) |
| 10 | Sin `SEX` | −0,0010 (−1,22; 2/15) | −0,0009 (−1,01; 3/15) | No mejora | Cuesta unos 0,001 de AUC |
| 11 | Sin `PAY_0` | −0,0295 (−6,03; 0/15) | −0,0340 (−5,79; 0/15) | No mejora | Dependencia fuerte del último estado de pago |

## Repetición con otros pliegues (paquete E)
Las semillas 7 y 99 son las mismas que mostraron el problema de la regla original, así que **no sirven como verificación independiente de la regla nueva**; solo muestran que la mejora del paquete se mantiene con otros pliegues.

| Semilla | LightGBM | Logística | Regla pareada | Regla original |
|---|---|---|---|---|
| 42 (la de la tabla) | +0,0017 | +0,0088 (pareada 5,00; mejora/ruido 1,49) | cumple | cumple |
| 7 | +0,0022 (1,53; 14/15) | +0,0088 (2,53; 15/15; 1,22) | cumple | cumple |
| 99 | +0,0029 (1,47; 14/15) | +0,0089 (3,30; 15/15; **0,81**) | cumple | **no cumple** |

## Métricas de control (paquete E frente a la referencia)
| Modelo | AUC | AUC-PR | Penalización relativa (menor es mejor) |
|---|---|---|---|
| LightGBM, referencia | 0,7853 | 0,5606 | 0,5471 |
| LightGBM, con E | 0,7870 | 0,5635 | 0,5456 |
| Logística, referencia | 0,7697 | 0,5440 | 0,5679 |
| Logística, con E | 0,7785 | 0,5483 | 0,5538 |

## Cómo se armó el paquete E
E se definió en una medición preliminar hecha con el mismo entrenamiento, antes de aplicar la regla. La regla se aplicó después **al paquete**, no a cada idea. Por eso `delay_streak`, `log_limit` y `pending_balance_last` están en E aunque solas «no pasen»: la columna «Decisión» de las ideas indica quién contribuye, no qué entra en E. Es una limitación de selección que se declara aquí.

## Conclusiones
1. **Paquete final: E.** Cinco variables simples, calculadas con datos conocidos al decidir y sin mirar el objetivo (`tests/test_features.py`). Constante `FINAL_FEATURES`.
2. **El beneficio es sobre todo para el modelo explicable.** La logística queda mucho más cerca de LightGBM (0,7785 frente a 0,7870).
3. **Para LightGBM la ganancia es pequeña** y no cambia la elección del modelo; eso lo decide el torneo (notebook 05).
4. **`PAY_0` domina el modelo:** sin ella el AUC cae unos 0,03. Debe figurar en el informe técnico.
5. **`SEX` aporta muy poco, pero no nada.** Quitarla cuesta unos 0,001 de AUC. Excluirla por equidad es una decisión del grupo, no de rendimiento.
6. **El paquete D** queda como alternativa si LightGBM gana el torneo.

## Qué se lleva el pipeline final
Las cinco variables de `FINAL_FEATURES` en `src/proyecto/features/build.py`: `utilization_mean`, `pending_balance_last`, `delay_streak`, `n_zero_payments` y `log_limit`. Se calculan dentro del `Pipeline` que arma `build_pipeline` en `src/proyecto/models/train.py`, así que la app puede recibir los datos crudos de un cliente y recalcularlas. Si LightGBM gana el torneo, conviene probar también el paquete D (`PACKAGE_D`).

## Supuestos y riesgos
- Las cinco variables solo valen si el banco tiene, al decidir, las seis facturas, los seis pagos y el límite del cliente. Es el mismo historial que ya usa el modelo, pero la app debe pedirlo completo.
- `utilization_mean`, `pending_balance_last` y `log_limit` dividen entre `LIMIT_BAL` o usan su logaritmo. Si el límite viene vacío o ≤ 0, la variable queda vacía y el `Pipeline` la imputa con la mediana de entrenamiento.
- `n_minimum_payments` (descartada) dependía de un 5 % supuesto.
- No hay riesgo de fuga detectado: ninguna variable usa el objetivo (`test_features_do_not_use_the_target`) y todo lo que se aprende de los datos se ajusta dentro de cada pliegue. La auditoría con el agente `auditor-de-fuga` no encontró fuga técnica (ver «Revisiones antes de subir»).

## Límites
- La regla pareada **no mira el tamaño** de la mejora: una mejora de +0,0007 puede pasar si es consistente. Por eso se confía más en el paquete que en cada idea.
- La regla se cambió después de ver un resultado (ver arriba).
- El paquete se eligió con los mismos datos con que se mide. La repetición con otros pliegues limita el optimismo, pero **no es una validación anidada**, que queda propuesta y pendiente de aprobación.
- Las ideas descartadas en una medición preliminar anterior (`max_atraso`, `atraso_medio`, `tendencia_pay`, razones de pago, tendencia de factura, `n_factura_neg`, `crece_deuda`) no se implementaron; sus cifras están en `especificaciones/04-features.md`, sección 6.
- `n_minimum_payments` usa un 5 % supuesto, no un dato.

## Revisiones antes de subir
- **Auditoría de fuga (agente `auditor-de-fuga`):** no encontró fuga técnica. Las variables usan solo la fila, el preprocesamiento va dentro del `Pipeline`, cada pliegue ajusta un `Pipeline` nuevo y la prueba no se abre. Señaló puntos de honestidad (ya incorporados arriba) y pruebas débiles (se agregaron tres: el ajuste usa solo las filas de entrenamiento, las variables dan lo mismo calculadas fila a fila, y la escala y las categorías se aprenden solo con entrenamiento).
- **Revisión de código (agente `revisor-de-codigo-ia`):** 142 pruebas en verde y sin secretos. Señaló como pendientes: los nombres de las variables están en español (la regla del proyecto pide inglés para el código), `decide` no tiene tolerancia en los bordes y los 15 pliegues comparten filas, por lo que el cociente pareado sale algo inflado.
- **Pendientes que siguen abiertos:** medir el paquete D con las semillas 7 y 99, la validación cruzada anidada y validar el 5 a 1 con el instructor.
