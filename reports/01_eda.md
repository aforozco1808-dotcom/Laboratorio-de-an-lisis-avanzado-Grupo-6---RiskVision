# Informe del EDA

> Resultado de `notebooks/02_eda.ipynb` (RiskVision, Grupo 6). Cada hallazgo termina en una decisión,
> que orienta limpieza y futuras hipótesis de modelado. Esta revisión documental no implementa cambios en el notebook 03.

**Alcance:** las descripciones, asociaciones, regla y auditoría de grupos usan el dataset completo, incluidas las filas de la prueba original. Solo el adelanto de modelos usa el 80 % original de entrenamiento. Se conservan las cifras guardadas sin recalcularlas. La prueba original ya fue explorada y evaluada; sus métricas no son una evaluación final completamente independiente. Las nuevas decisiones se tomarán solo mediante validación dentro del entrenamiento original; se propone validación cruzada anidada, reconociendo que no elimina la exposición previa. Una evaluación final independiente requeriría datos nuevos no utilizados para decidir.

## Qué son los datos

- **Filas × columnas:** 30.000 × 25 (24 variables más `ID`).
- **Unidad de análisis (qué es una fila):** un cliente de tarjeta de crédito de un banco de Taiwán (abril a septiembre de 2005).
- **¿Una entidad aparece en varias filas?** No. Hay 30.000 `ID` distintos, uno por fila.
- **Objetivo y distribución:** `default payment next month`; 22,12 % incumple (6.636) y 77,88 % paga (23.364). Sin vacíos.
- **Línea base** (resultado de una regla trivial): «siempre paga» tiene exactitud 77,88 %, recall 0 % y ROC AUC 0,5000.

## Calidad de los datos

- Sin vacíos y sin texto: las 25 columnas son enteras. Sin duplicados completos; 35 filas idénticas si se ignora el `ID` (no se borran).
- `EDUCATION` tiene códigos sin documentar: 0 (14), 5 (280) y 6 (51), un total de 345 clientes (1,15 %). `MARRIAGE` tiene el código 0 en 54 clientes.
- `PAY_*` mezcla estados (-2, -1, 0) con meses de atraso (1 a 8); no existe `PAY_1` (se salta de `PAY_0` a `PAY_2`).
- 1.930 clientes (6,43 %) tienen algún `BILL_AMT` negativo; hay colas largas en `PAY_AMT*` (en `PAY_AMT2`, percentil 99 = 76.651 y máximo = 1.684.259).
- Las 6 facturas mensuales están muy correlacionadas entre sí (0,89 en promedio).
- 3.931 clientes facturan por encima de su límite en algún mes y 795 no tienen movimiento en 6 meses (plausibles).
- Línea base de negocio: la regla «marcar si `PAY_0` ≥ 2» marca al 10,4 %, tiene recall 32,8 % y precision 69,6 %, y su penalización relativa bajo la hipótesis económica 5 a 1 del grupo es 0,775; no es un costo observado del banco ni dinero.
- Línea base de equidad (misma regla): recall 32,7 % en hombres y 32,9 % en mujeres; entre 30 % y 35 % según la edad.
- Sin nombres ni documentos (solo un `ID` anónimo). `SEX`, `AGE` y `MARRIAGE` son atributos sensibles.

## Asociaciones y límites del proceso

**Columnas y proceso:** `PAY_*`, `BILL_AMT*` y `PAY_AMT*` cubren abril a septiembre, antes del mes objetivo; no se identifica una fuga temporal evidente, condicionada a que estén disponibles al decidir. `PAY_0` es la variable original más correlacionada con el objetivo (0,33), no necesariamente la más importante en un modelo. Los estados -2, -1 y 0 muestran aproximadamente 13–17 % de incumplimiento; el código 2, 69,1 %, pero desde ese código las tasas no aumentan estrictamente (ver la tabla de PAY_0 en la sección 3.3 del notebook). Son asociaciones, no causas.

Se conserva la propuesta de comparar modelos con y sin `PAY_0` mediante validación dentro del entrenamiento; esto no sustituye la revisión temporal. El EDA global ya consultó la prueba original y el notebook de selección ya la evaluó. En adelante no se usará para seleccionar variables, modelo, calibración ni umbral, y no se presentará como evaluación final independiente.

## Variables candidatas

**Lectura exploratoria sobre el dataset completo:**
- Las 6 facturas están **muy correlacionadas** (0,89 en promedio), pero no son idénticas. Esto motiva probar resúmenes o regularización sin eliminar información automáticamente.
- **`n_atraso` es candidata por asociación:** su correlación con el objetivo (0,40) supera a la de `PAY_0` (0,33). La tasa observada va de 11,7 % con 0 meses codificados con atraso a 70,3 % con 6. No demuestra causalidad, mejora incremental ni que sea la mejor variable nueva.
- `ratio_pago` (−0,01) y `tendencia` (−0,03) tienen poca asociación lineal con el objetivo. Esto no permite anticipar su utilidad en relaciones no lineales o interacciones.
- El código calcula `ratio_pago` como suma de pagos dividida por suma de facturas truncadas a cero; si el denominador es cero genera un vacío. Es un resumen de saldos mensuales, no una fracción exacta de deuda liquidada. Antes de incorporarlo al pipeline hay que definir el manejo de vacíos y extremos. Todas las candidatas se compararán dentro del entrenamiento original.

## Comparación exploratoria y calibración

**Lectura:**
- **Promedios próximos en esta exploración:** LightGBM 0,7813 e HistGradientBoosting 0,7806; como referencia histórica, Gradient Boosting obtuvo 0,7815. XGBoost inicial queda en 0,7668 y regresión logística en 0,7272. La diferencia entre los dos primeros es pequeña respecto a la dispersión entre folds (≈ 0,006), pero no demuestra empate estadístico. La elección requiere evaluación comparable de configuraciones y diferencias pareadas dentro del entrenamiento.
- **Indicios de calibración agregada en HistGradientBoosting:** el quintil superior tiene media predicha 0,568 y observada 0,567; el Brier 0,1339 mejora la referencia constante 0,1723. Brier no mide exclusivamente calibración y cinco quintiles pueden ocultar desajustes locales. No garantiza calibración cerca de `1/6 ≈ 0,167` ni en Gradient Boosting optimizado; debe comprobarse el modelo concreto.
- **Ganancia pequeña en el tramo observado:** de 14.400 a 19.200 filas el AUC aumenta 0,0009. Sin dispersión, repetición ni tamaños mayores no se puede concluir que más datos ayuden poco ni que mejores variables garanticen mejoras.
- **Candidatos para el torneo:** conservar gradient boosting y regresión logística como contraste interpretable, sin declarar ganador definitivo ni utilidad económica. SHAP puede describir contribuciones a predicciones; no explica causas ni demuestra el efecto de intervenir.

## Regla y capacidad

**Lectura exploratoria sobre el dataset completo:**
- La regla marca al 10,4 % y presenta precision 69,6 % y recall 32,8 %: detecta aproximadamente 1 de cada 3 incumplidores.
- Su penalización media (0,775) es próxima a marcar a todos (0,779). El 0,57 del modelo en el notebook 01 se conserva como referencia exploratoria pendiente de trazabilidad. No se evaluaron aquí sobre las mismas observaciones; no demuestra superioridad validada ni ahorro económico. La afirmación previa de ahorro cercano al 27 % se retira como conclusión demostrada.
- Con el orden usado, el 5–10 % inicial tiene lift 3,2–3,3; al revisar 20 %, el lift es 2,4 y se captura el 48,5 % de los incumplidores. Son resultados descriptivos, sensibles a empates en el orden y a esta muestra; no prueban impacto de contactar clientes ni desempeño futuro.
- Con matriz `[[0, 1], [5, 0]]`, el umbral teórico es `1/6 ≈ 0,167` solo para probabilidades calibradas y sin otros costos o restricciones. `PAY_0 >= 2` es una regla sobre un código, no ese umbral de probabilidad. El umbral operativo se seleccionará posteriormente solo mediante validación dentro del entrenamiento original.

## Equidad

**Lectura:** el recall observado es parecido en hombres y mujeres (32,7 % y 32,9 %), mientras sus tasas de incumplimiento son 24,2 % y 20,8 %. Por edad el recall varía entre 30 % y 35 %. Esto no demuestra equidad: faltan otras tasas de error, precision e incertidumbre, y el análisis usa el dataset completo. Son asociaciones descriptivas, no explicaciones causales.

`MARRIAGE` tiene grupos de 54 y 323 clientes en la tabla de categorías, pero aquí no se calcularon sus errores. La auditoría posterior debe usar la agrupación definitiva, incluir soporte y comparar falsos positivos, precision y recall con incertidumbre. No se inventan esos resultados en esta revisión.

## Decisiones

| # | Hallazgo (con cifra) | Decisión | Motivo |
|---|---|---|---|
| 1 | 30.000 ID distintos, uno por fila | Mantener la partición 80/20, estratificada, semilla 42, y excluir el ID | Ningún ID queda en dos partes; no cambia que la prueba ya se vio |
| 2 | 22,12 % incumple; «siempre paga»: exactitud 77,88 %, recall 0 % | Usar ROC AUC, recall y AUC-PR; medir también la penalización relativa y la carga de revisión | La exactitud sola no mide la detección ni la hipótesis 5 a 1 |
| 3 | EDUCATION 0, 5 y 6: 345 clientes (1,15 %); MARRIAGE 0: 54 | Agrupar en «otros» (4 y 3) antes de codificar, con la misma regla en entrenamiento e inferencia | Es una convención, no un significado demostrado; hay que integrarla al pipeline |
| 4 | PAY_0: estados -2, -1 y 0 con 13–17 %; código 2, 69,1 %; código 5, 50,0 % | Dejar los códigos como números por ahora y comparar otras codificaciones con validación | Mezclan estados y meses de atraso; la relación no es causal ni siempre creciente; hay que documentar los códigos |
| 5 | 1.930 clientes (6,43 %) con BILL_AMT negativo; PAY_AMT2: p99 76.651, máximo 1.684.259 | Conservar los negativos plausibles; usar log1p solo donde se pueda; comparar Yeo-Johnson dentro del pipeline | log y log1p no sirven con todos los negativos; cualquier transformación se aprende solo con entrenamiento |
| 6 | Mujeres: tasa 20,8 %, recall de la regla 32,9 %; hombres: 24,2 % y 32,7 % | Probar con y sin SEX y revisar errores por sexo, edad y estado civil | Un recall parecido no demuestra equidad; faltan otras métricas e incertidumbre |
| 7 | PAY_0: correlación 0,33 con el objetivo | Comparar el modelo con y sin PAY_0 y confirmar que se conoce al decidir | Mide cuánto depende el modelo de ella; no prueba causalidad ni ausencia de fuga |
| 8 | 35 filas idénticas sin ID; 3.931 clientes sobre su límite | Conservar las filas y documentar que son plausibles | Los ID son distintos y las coincidencias no prueban errores |
| 9 | n_atraso: correlación 0,40 frente a 0,33 de PAY_0; tasas de 11,7 % a 70,3 % | Conservarla como candidata y medir con validación si mejora el modelo | Es una asociación; no demuestra que aporte algo nuevo |
| 10 | Facturas: correlación media 0,89; ratio_pago −0,01 y tendencia −0,03 con el objetivo | Comparar resúmenes o regularización, y las candidatas; definir cómo tratar vacíos y extremos del ratio | Una correlación lineal baja no descarta una relación más compleja ni obliga a eliminar variables |
| 11 | Regla: penalización 0,775; marcar a todos: 0,779; referencia exploratoria del modelo: 0,57 | Comparar las alternativas con las mismas observaciones de validación y la misma matriz | No demuestra un ahorro del 27 % ni impacto económico; el 0,57 está pendiente de verificar |
| 12 | LightGBM 0,7813; HistGradientBoosting 0,7806; XGBoost 0,7668; logística 0,7272 | Mantener boosting y logística como candidatos, sin ganador | Son promedios exploratorios; falta comparar con diferencias pareadas y configuraciones ajustadas |
| 13 | HistGradientBoosting: quintil superior predicho 0,568 y observado 0,567 | Revisar la calibración del modelo concreto cerca del umbral y elegir el umbral solo con validación | La evidencia es preliminar y agregada; no garantiza calibración cerca de 1/6 ≈ 0,167 |
| 14 | De 14.400 a 19.200 filas: el AUC sube 0,0009 | Dejar abiertas las dos hipótesis: más datos y mejores variables | Sin variación ni repeticiones no se puede decir que ya se saturó ni que habrá mejoras |

## Unidad de análisis y partición

Cada fila es un cliente distinto (30.000 `ID` únicos), así que **partir por fila es válido**: ningún ID queda en ambas particiones si se cumplen las comprobaciones; la unicidad del identificador no demuestra por sí sola independencia real entre personas ni generalización temporal. Se estratifica por el objetivo (22,12 % incumple) con 80 % entrenamiento (24.000) y 20 % prueba (6.000), semilla 42. `tests/test_split.py` contiene comprobaciones de partición y estratificación; no se ejecutaron en esta revisión documental. Implementación: `src/proyecto/data/split.py`.

## Qué queda abierto

- Contrastar la hipótesis económica 5 a 1. El umbral teórico 1/6 ≈ 0,167 depende de probabilidades calibradas y de la matriz asumida, sin otros costos o restricciones. El umbral posterior se seleccionará exclusivamente mediante validación dentro del entrenamiento original.
- La comparación oficial de modelos (con optimización de cada candidato) se hace en `05_torneo.ipynb`.
- Medir el modelo con y sin `PAY_0` y con y sin `SEX` (etapa 4). **Hecho en `reports/04_features.md`:** sin `PAY_0` el AUC cae unos 0,03 y sin `SEX` baja unos 0,001. Allí también se midieron `n_atraso` y las demás variables candidatas: `n_atraso` no mejora ningún modelo.
- Documentar el significado de los códigos de pago y comparar sus codificaciones; no asumir que el orden numérico implica riesgo estrictamente creciente.
- Revisar posteriormente la integración de limpieza fija, imputación, codificación y transformaciones en el pipeline, conservando el tratamiento justificado de negativos y filas. log1p solo admite valores mayores que -1; Yeo-Johnson es una candidata para montos con negativos.
- Recalcular comparaciones homogéneas solo en una fase autorizada. La prueba original se reportará como complementaria; cambiar la semilla no proporciona datos independientes.
