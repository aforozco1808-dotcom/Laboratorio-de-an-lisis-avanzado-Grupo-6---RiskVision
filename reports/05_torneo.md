# Informe del torneo de modelos

> Resultado de `notebooks/05_torneo.ipynb` (RiskVision, Grupo 6). Especificación: `especificaciones/05-torneo.md`.
> El torneo **descarta**: elige 1–2 finalistas con evidencia, no un campeón. Resultado por pliegue: `reports/05_torneo_por_pliegue.csv`.

## Reglas del torneo
- **Mismos datos y mismos pliegues para todos:** 24.000 clientes de entrenamiento, 15 pliegues (5 estratificados × 3 repeticiones, semilla 42, los mismos de la etapa 4), partidos por fila porque los 30.000 `ID` son únicos. La prueba no se abrió y, por la revisión H1, no es una evaluación final independiente.
- **Líneas base (tres):** «siempre paga» (acierta 77,9 % sin detectar a nadie), «prior» (misma probabilidad para todos; con el umbral 1/6 equivale a marcar a todos) y la regla simple del EDA `PAY_0 ≥ 2`.
- **Métrica principal:** ROC AUC, como dice `reports/00_caso.md`. **Control:** AUC-PR, log loss, Brier, exactitud balanceada, F1 macro, precisión y recall por clase, penalización relativa con umbral fijo (1/3, 1/6 y 1/11 según el escenario 2:1, 5:1 y 10:1) y recall/lift al revisar el 5 %, 10 % y 20 % de mayor riesgo. El 5 a 1 es una hipótesis del grupo, no un costo observado; las capacidades de revisión son supuestos.
- **Variables:** el paquete final de la etapa 4 (`utilization_mean`, `pending_balance_last`, `delay_streak`, `n_zero_payments`, `log_limit`) para todos. Variantes declaradas: LightGBM con el paquete D (10 variables) y LightGBM calibrado (calibración isotónica con validación interna, 3 pliegues).
- **Modelos con parámetros fijos, sin optimizar:** ver `especificaciones/05-torneo.md`. Sin pesos de clase.

## Tabla comparativa
Media ± desviación entre pliegues, ordenada por ROC AUC.

| Modelo | ROC AUC (± entre pliegues) | Penalización 5:1 | Segundos por ajuste |
|---|---|---|---|
| LightGBM + paquete D | 0,7885 ± 0,0060 | 0,541 | 0,26 |
| LightGBM calibrado | 0,7877 ± 0,0064 | 0,543 | 0,68 |
| XGBoost | 0,7872 ± 0,0063 | 0,549 | 0,44 |
| LightGBM | 0,7870 ± 0,0063 | 0,546 | 0,26 |
| bosque aleatorio | 0,7868 ± 0,0057 | 0,548 | 1,72 |
| regresión logística | 0,7785 ± 0,0059 | 0,554 | 0,16 |
| línea base: regla PAY_0 ≥ 2 | 0,6435 ± 0,0058 | 0,775 | 0,00 |
| línea base: prior | 0,5000 ± 0,0000 | 0,779 | 0,04 |
| línea base: siempre paga | 0,5000 ± 0,0000 | 1,106 | 0,03 |

## Métricas técnicas
Media entre pliegues. Las de decisión usan «marcar si la probabilidad ≥ 1/6»; la tasa real de incumplimiento es 0,221. Las dos reglas que dan solo 0 o 1 no tienen log loss, Brier ni probabilidad media interpretables.

| Modelo | AUC | AUC-PR | Exactitud balanceada | F1 macro | Precisión (incumple) | Recall (paga) | Recall (incumple) | % marcado | Log loss | Brier | Prob. media |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM + paquete D | 0,789 | 0,563 | 0,704 | 0,633 | 0,382 | 0,654 | 0,755 | 0,437 | 0,424 | 0,133 | 0,221 |
| LightGBM calibrado | 0,788 | 0,562 | 0,702 | 0,628 | 0,378 | 0,644 | 0,759 | 0,445 | 0,426 | 0,133 | 0,221 |
| XGBoost | 0,787 | 0,564 | 0,701 | 0,631 | 0,380 | 0,654 | 0,747 | 0,434 | 0,424 | 0,133 | 0,221 |
| LightGBM | 0,787 | 0,563 | 0,702 | 0,633 | 0,382 | 0,656 | 0,749 | 0,434 | 0,425 | 0,133 | 0,221 |
| bosque aleatorio | 0,787 | 0,559 | 0,696 | 0,616 | 0,366 | 0,619 | 0,772 | 0,467 | 0,426 | 0,134 | 0,222 |
| regresión logística | 0,779 | 0,548 | 0,705 | 0,648 | 0,400 | 0,695 | 0,714 | 0,395 | 0,431 | 0,135 | 0,221 |
| línea base: regla PAY_0 ≥ 2 | 0,643 | 0,377 | 0,643 | 0,669 | 0,697 | 0,960 | 0,327 | 0,104 | — | — | — |
| línea base: prior | 0,500 | 0,221 | 0,500 | 0,181 | 0,221 | 0,000 | 1,000 | 1,000 | 0,528 | 0,172 | 0,221 |
| línea base: siempre paga | 0,500 | 0,221 | 0,500 | 0,438 | 0,000 | 1,000 | 0,000 | 0,000 | — | — | — |

**Sobre el CSV por pliegue.** `reports/05_torneo_por_pliegue.csv` conserva log loss, Brier y probabilidad media también para las reglas que dan solo 0 o 1 («siempre paga» y `PAY_0 ≥ 2`); esos valores no tienen sentido y por eso las tablas los ocultan.

## Sensibilidad a los supuestos
**Cuánto pesa no detectar a quien incumple** (penalización por cliente, menor es mejor; cada escenario con su umbral teórico):

| Modelo | Penalización 2:1 | Penalización 5:1 | Penalización 10:1 |
|---|---|---|---|
| LightGBM + paquete D | 0,299 | 0,541 | 0,686 |
| LightGBM calibrado | 0,299 | 0,543 | 0,692 |
| XGBoost | 0,299 | 0,549 | 0,690 |
| LightGBM | 0,300 | 0,546 | 0,691 |
| bosque aleatorio | 0,300 | 0,548 | 0,686 |
| regresión logística | 0,304 | 0,554 | 0,714 |
| línea base: regla PAY_0 ≥ 2 | 0,329 | 0,775 | 1,519 |
| línea base: prior | 0,442 | 0,779 | 0,779 |
| línea base: siempre paga | 0,442 | 1,106 | 2,212 |

**Cuántos clientes se pueden revisar** (fracción de incumplidores capturada al revisar solo a los de mayor riesgo):

| Modelo | Captura al revisar 5 % | Captura al revisar 10 % | Captura al revisar 20 % | Lift 20 % |
|---|---|---|---|---|
| LightGBM + paquete D | 0,172 | 0,319 | 0,516 | 2,58 |
| LightGBM calibrado | 0,171 | 0,320 | 0,517 | 2,59 |
| XGBoost | 0,172 | 0,320 | 0,518 | 2,59 |
| LightGBM | 0,171 | 0,319 | 0,516 | 2,58 |
| bosque aleatorio | 0,170 | 0,319 | 0,516 | 2,58 |
| regresión logística | 0,170 | 0,317 | 0,510 | 2,55 |
| línea base: regla PAY_0 ≥ 2 | 0,158 | 0,315 | 0,400 | 2,00 |
| línea base: prior | 0,050 | 0,100 | 0,200 | 1,00 |
| línea base: siempre paga | 0,050 | 0,100 | 0,200 | 1,00 |

- **Los árboles apenas cambian entre escenarios:** el mejor es LightGBM calibrado o la variante D con 2 a 1, la variante D con 5 a 1 y la variante D con el bosque con 10 a 1, siempre con diferencias de 0,001 a 0,008 entre ellos.
- **La logística pierde más cuanto más cuesta no detectar:** su diferencia con el mejor es de 0,005 con 2 a 1, 0,013 con 5 a 1 y 0,028 con 10 a 1.
- **Con poca capacidad de revisión casi no hay diferencia:** al revisar solo el 5 % todos capturan 17,0–17,2 %.

## Los modelos frente a la regla simple `PAY_0 ≥ 2`
La regla marca al 10,4 % de los clientes, con AUC 0,643, precisión 0,697 y recall 0,327. Su penalización 5:1 (0,775) es casi igual a marcar a todos (0,779).
- **Los modelos la superan con claridad:** AUC 0,787 frente a 0,643; penalización 0,54–0,55 frente a 0,775; y al revisar el 20 %, 51–52 % de incumplidores capturados frente a 40,0 %.
- **Pero en los primeros tramos casi no pierde:** al revisar el 5 % o el 10 %, la regla captura 15,8 % y 31,5 %, y los modelos 17,0–17,2 % y 31,7–32,0 %. El valor del modelo está en ordenar bien al resto de los clientes.
- **Empates:** la regla da 0 o 1, así que muchos clientes tienen la misma probabilidad. En el corte de capacidad se reparte el cupo por valor esperado (cada cliente empatado aporta la fracción que le toca), por lo que la cifra no depende del orden de las filas.

## Calibración por tramos
Predicciones de los 15 pliegues juntas (cada cliente aparece 3 veces, una por repetición, y esas copias no son independientes), en hasta 10 grupos según la probabilidad predicha; clientes con la misma probabilidad quedan siempre juntos. «Error de calibración» es la diferencia media entre lo predicho y lo observado (0 = perfecta). La zona del umbral es probabilidad entre 12 % y 22 %.

| Modelo | Error de calibración | Predicciones con prob. 12–22 % | Prob. predicha ahí | Tasa observada ahí |
|---|---|---|---|---|
| LightGBM | 0,0038 | 21.800 | 0,162 | 0,161 |
| LightGBM calibrado | 0,0038 | 23.896 | 0,165 | 0,164 |
| LightGBM + paquete D | 0,0041 | 21.035 | 0,163 | 0,162 |
| XGBoost | 0,0053 | 22.866 | 0,162 | 0,162 |
| bosque aleatorio | 0,0092 | 24.361 | 0,164 | 0,158 |
| regresión logística | 0,0101 | 23.558 | 0,157 | 0,169 |

- **LightGBM está entre los mejor calibrados** (empatado con su versión calibrada y muy cerca de la variante D y de XGBoost; las diferencias de 0,0003 a 0,0015 no se distinguen del ruido). En la zona del umbral predice 16,2 % y se observa 16,1 %, aunque dentro de esa zona hay tramos que se compensan (por ejemplo, 19,9 % predicho y 18,3 % observado en el 7.º tramo). Eso no contradice usar 1/6 con este modelo.
- **La logística subestima el riesgo ahí** (15,7 % predicho frente a 16,9 % observado) y **el bosque lo sobrestima** (16,4 % frente a 15,8 %).
- **Calibrar LightGBM con regresión isotónica no mejora nada:** mismo error de calibración, tres veces más lento. La versión calibrada es un promedio de 3 modelos entrenados con dos tercios de los datos, así que su diferencia mezcla calibrar y promediar. La calibración se revisó en promedio y por 10 tramos; no se hizo por subgrupos (sexo, edad).

## ¿Coinciden los rankings?
- Entre los modelos, por **AUC** lidera LightGBM con el paquete D; por **AUC-PR** y por **recall al revisar el 20 %**, XGBoost; por **F1 macro**, la regresión logística. Las diferencias entre los modelos de árboles son mínimas (AUC de 0,7868 a 0,7885, con ±0,006 entre pliegues): es un empate práctico.
- **Cómo deciden:** los árboles marcan 43–47 % de los clientes y detectan 75–77 % de los que incumplen; la logística marca 39,5 % y detecta 71 %, con mejor precisión. El bosque es el que más marca (46,7 %) y el que deja pasar a menos clientes que pagan (recall de «paga» 0,62).
- **Con capacidad fija casi no hay diferencia:** al revisar el 20 % de mayor riesgo todos capturan 51,0–51,8 % de los incumplidores.
- **Ningún modelo sacrifica una clase:** recall de «paga» 0,62–0,70 y de «incumple» 0,71–0,77.

## Comparación pareada
Misma medida de la etapa 4: diferencia media ÷ variación de la diferencia entre pliegues (mayor que 1 es consistente). Siempre aparece primero el de mayor AUC medio.

| Mejor | Contra | Diferencia media de AUC | Pliegues que gana | Diferencia pareada |
|---|---|---|---|---|
| LightGBM + paquete D | LightGBM | 0,0015 | 14 de 15 | 1,26 |
| LightGBM + paquete D | XGBoost | 0,0013 | 11 de 15 | 0,77 |
| XGBoost | LightGBM | 0,0002 | 8 de 15 | 0,15 |
| LightGBM | bosque aleatorio | 0,0001 | 8 de 15 | 0,07 |
| LightGBM calibrado | LightGBM | 0,0007 | 10 de 15 | 0,62 |
| LightGBM | regresión logística | 0,0084 | 15 de 15 | 3,45 |

**Paquete D frente a E en LightGBM, con pliegues nuevos** (semillas 7 y 99): +0,0013 y +0,0012 de AUC, gana en 12 de 15 pliegues en ambas y la diferencia pareada es 0,78 y 0,74. En los pliegues de la etapa 4 había salido +0,0015, 14 de 15 y 1,26. La ventaja de D sigue siendo positiva pero pequeña, y con pliegues nuevos no alcanza el umbral de 1; no compensa pasar de 5 a 10 variables, así que no se adopta. Las semillas 7 y 99 reparten otra vez los mismos 24.000 clientes con los que se eligió D: reducen el sesgo de selección, pero no lo eliminan. Los 15 pliegues comparten filas, por lo que estas diferencias pareadas pueden salir algo infladas.

## ¿Cuánto vale la diferencia?
Por cada 1.000 clientes (unos 221 incumplen), con el umbral 1/6. No se estima un volumen mensual: el grupo no tiene un dato del volumen del banco. Son conteos y penalizaciones bajo la hipótesis 5 a 1, no dinero.

| Modelo | Se marcan | Detectados | Se escapan | Falsas alarmas | Penalización total 5:1 | Detectados al revisar 20 % |
|---|---|---|---|---|---|---|
| LightGBM + paquete D | 437 | 167 | 54 | 270 | 541 | 114 |
| LightGBM calibrado | 445 | 168 | 53 | 277 | 543 | 114 |
| XGBoost | 434 | 165 | 56 | 269 | 549 | 115 |
| LightGBM | 434 | 166 | 55 | 268 | 546 | 114 |
| bosque aleatorio | 467 | 171 | 50 | 296 | 548 | 114 |
| regresión logística | 395 | 158 | 63 | 237 | 554 | 113 |
| línea base: regla PAY_0 ≥ 2 | 104 | 72 | 149 | 32 | 775 | 88 |
| línea base: prior | 1000 | 221 | 0 | 779 | 779 | 44 |
| línea base: siempre paga | 0 | 0 | 221 | 0 | 1106 | 44 |

- **Frente a marcar a todos** (penalización 779 por cada 1.000), los modelos bajan a 541–554, revisando 40–47 % de los clientes en vez de todos.
- **Entre el mejor y la logística:** 13 puntos de penalización por cada 1.000 clientes con 5 a 1 (541 frente a 554) y 1 incumplidor menos capturado al revisar el 20 % (113 frente a 114).
- **Redondeo:** los conteos por cada 1.000 clientes están redondeados y pueden diferir en 1 entre el notebook y esta tabla.
- **Líneas base:** en «siempre paga» y «prior» todas las probabilidades son iguales, así que el cálculo da exactamente lo esperado por azar: 44 de 221 (20 %), con lift 1,0.

## Quién ganó y finalistas
**Gana LightGBM con el paquete E, por desempate y no por tener el AUC más alto.** El AUC no separa a los modelos de árboles (ninguna pareja es consistente: XGBoost gana a LightGBM en 8 de 15 pliegues con diferencia pareada 0,15, y LightGBM al bosque en 8 de 15 con 0,07), así que decide el criterio siguiente de la especificación: calibración, tiempo y simplicidad. LightGBM está entre los mejor calibrados, ajusta en 0,26 s (XGBoost 0,44; calibrado 0,68; bosque 1,7) y usa solo 5 variables nuevas.

| Finalista | Por qué (consistencia, métrica, tiempo, interpretabilidad) | Qué podría cambiar con la optimización |
|---|---|---|
| **LightGBM con el paquete E** | Empata en AUC con XGBoost, el bosque y la variante D, pero es el más rápido, está bien calibrado en la zona del umbral y supera con claridad a la regla simple | XGBoost o el bosque podrían igualarlo o pasarlo al afinar. Umbral y calibración se deciden en la etapa 6 |
| **Regresión logística con el paquete E** | Contraste explicable. LightGBM le gana en 15 de 15 pliegues (+0,0084 de AUC, diferencia pareada 3,5; con el paquete D, +0,0100 y 4,8), pero al revisar el 20 % captura casi el mismo número de incumplidores (113 frente a 114 por cada 1.000 clientes) | Su desventaja crece con el costo de no detectar (ver sensibilidad) y subestima el riesgo cerca del umbral, por lo que habría que calibrarla o usar otro umbral. Probablemente sea la que menos gane al afinarse (hipótesis, no medido) |

**Modelos descartados y por qué**
- **XGBoost:** empata con LightGBM, es casi el doble de lento y sus ventajas (≤ 0,002 en AUC-PR y en el recall al revisar el 20 %) están dentro del ruido.
- **Bosque aleatorio:** mismo AUC, siete veces más lento, el que más clientes marca y el peor calibrado de los árboles.
- **LightGBM con el paquete D:** la ventaja es pequeña y no llega a 1 con pliegues nuevos; exige 10 variables en vez de 5.
- **LightGBM calibrado:** igual de bien calibrado que el original, tres veces más lento y sin mejora clara.
- **Líneas base:** no son candidatos; solo fijan el piso.

**Por qué el torneo no elige un campeón.** Cuatro modelos empatan dentro del ruido y ninguno está afinado, así que al optimizar el orden puede cambiar. Además, nada de aquí es una evaluación final, porque la prueba original ya se vio antes.

## Comparación con el notebook de selección de modelo
`model_selection_classification_REVISION_FINAL.ipynb` (selección y optimización previas del grupo, octubre) se comparó con este torneo **en los mismos 15 pliegues** y con el mismo `Pipeline` de preprocesamiento. Las cifras salen de scripts exploratorios que **no están en el repositorio**; la afinación formal se hará en la etapa 6.

**En qué se diferencian**

| | Notebook de selección de modelo | Este torneo |
|---|---|---|
| Candidatos | 5 (logística, KNN, árbol, Gradient Boosting, línea base) | 4 modelos, 2 variantes y 3 líneas base, con la regla `PAY_0 ≥ 2` |
| Pliegues | 5, una repetición | 15 (5 × 3), iguales para todos |
| Comparación | Solo la media | Pareada: en cuántos pliegues gana y qué tan consistente es |
| Variables | Las 23 originales | Más un paquete de 5 variables nuevas |
| `PAY_*` en la logística | Como número | Como estados |
| Métricas | Solo ROC AUC | AUC, AUC-PR, penalización, recall por clase, calibración |
| Supuestos | No los prueba | Sensibilidad al 5 a 1 y a la capacidad de revisión |
| Afinación | Optuna, 30 pruebas, solo Gradient Boosting | No afinado todavía |
| Reproducibilidad | Ruta local fija e instalación de paquetes dentro del notebook | Rutas del proyecto y entorno del repositorio |

**Lo que se midió en los mismos 15 pliegues** (cifras de scripts exploratorios, no reproducibles desde el repositorio)

| Modelo | AUC (± entre pliegues) | Segundos por ajuste |
|---|---|---|
| Gradient Boosting por defecto (original) | 0,7817 ± 0,0075 | 4,9 |
| Gradient Boosting optimizado con Optuna (original) | 0,7843 ± 0,0070 | 12,9 |
| Regresión logística con `PAY_*` como número (original) | 0,7273 ± 0,0101 | 0,05 |
| Regresión logística con `PAY_*` como estados, sin variables nuevas | 0,7697 ± 0,0060 | 0,12 |
| LightGBM sin variables nuevas | 0,7853 ± 0,0071 | 0,21 |
| LightGBM con el paquete E (este torneo) | 0,7870 | 0,26 |
| LightGBM con el paquete D (este torneo) | 0,7885 | 0,26 |
| Regresión logística con el paquete E (este torneo) | 0,7785 | 0,16 |

| Comparación (A frente a B) | Diferencia de AUC | A gana en | Diferencia pareada |
|---|---|---|---|
| LightGBM (paquete E) frente al GB optimizado original | +0,0026 | 13 de 15 | 1,40 |
| LightGBM (paquete D) frente al GB optimizado original | +0,0042 | 15 de 15 | 2,00 |
| LightGBM sin variables nuevas frente al GB optimizado original | +0,0010 | 12 de 15 | 0,65 |
| GB optimizado frente al GB por defecto (original) | +0,0027 | 13 de 15 | 1,20 |
| Logística con `PAY_*` como estados frente a como número | +0,0424 | 15 de 15 | 6,36 |
| GB optimizado original frente a nuestra logística | +0,0058 | 14 de 15 | 2,29 |

**Qué se valida y qué no**
- **Bien hecho:** parte los datos antes de preprocesar, mete el preprocesamiento en el `Pipeline` y fijó la regla de selección antes de ver resultados.
- **La conclusión sobre la regresión logística es un artefacto.** Tratar `PAY_*` como números da 0,727; como estados sube a 0,770 (+0,042, en los 15 pliegues). Con las variables nuevas llega a 0,7785, a solo unas milésimas del boosting. Los textos que citan «logística 0,73» (notebooks 01 y 02, `especificacion-proyecto.md`) vienen de ese resultado.
- **El 0,7850 del tuning es optimista:** es el mejor de 30 pruebas sobre los mismos pliegues. En los pliegues de este torneo, ese mismo modelo da 0,7843. (Ese notebook cita 0,7784 en la prueba; es una cifra citada, no verificada aquí, y esa prueba ya no es independiente, así que no se usa como argumento.)
- **El tuning aportó poco:** +0,0027 de AUC, con diferencia pareada de 1,2.
- **Cobertura limitada:** sin bosque, LightGBM ni XGBoost, y solo uno de los modelos se optimizó.
- **Nuestro LightGBM, sin afinar, iguala o supera a su Gradient Boosting afinado** y ajusta unas 50 veces más rápido. La ventaja parece venir de las variables nuevas, pero esas variables se eligieron en estos mismos pliegues, por lo que puede estar algo inflada.

**Afinación exploratoria de LightGBM (paquete E)** (script exploratorio, no reproducible desde el repositorio)
Optuna, 60 pruebas en 160 s, buscando en pliegues de semilla 123 y evaluando en pliegues que la búsqueda no vio. Los 24.000 clientes son los mismos, así que reduce el sesgo de selección pero no lo elimina.

| Semilla de evaluación | AUC original | AUC afinado | Mejora | Pliegues | Diferencia pareada |
|---|---|---|---|---|---|
| 42 | 0,7870 | 0,7884 | +0,0014 | 13 de 15 | 0,87 |
| 7 | 0,7871 | 0,7880 | +0,0009 | 12 de 15 | 0,78 |
| 99 | 0,7876 | 0,7886 | +0,0010 | 12 de 15 | 0,80 |

- **La mejora (≈ +0,001) no es consistente:** la diferencia pareada no llega a 1 y la penalización 5:1 casi no cambia (0,5456 a 0,5433 en la semilla 42).
- **Parámetros encontrados (punto de partida para la etapa 6, no definitivos):** `learning_rate=0.014`, `n_estimators=388`, `num_leaves=26`, `min_child_samples=91`, `subsample=0.53`, `colsample_bytree=0.89`, `reg_lambda=8.96`, `reg_alpha=0.06`. Ninguno quedó en el borde del rango de búsqueda.
- **Mezclar LightGBM afinado (80 %) con la logística (20 %)** sube el AUC solo +0,0005 a +0,0007 (diferencia pareada de 0,66 a 1,20); con 50/50 empeora. Solo se probaron esos dos pesos, sin validación previa. No justifica la complejidad.
- **Lectura:** con estos datos y variables el techo parece estar cerca de 0,789 de AUC. Los avances más probables están en la decisión (umbral, calibración de la logística) y no en el AUC.

## Efectos que la métrica principal no ve
- **Carga de revisión:** los árboles piden revisar 43–47 % de los clientes con el umbral 1/6; la logística, 39,5 %. Si el equipo de riesgo no puede revisar tanto, el umbral operativo será otro y habrá que decidirlo en la etapa 6 con validación.
- **El umbral 1/6 favorece a los modelos bien calibrados,** y por eso se agregaron la captura con capacidad fija y la revisión por tramos.
- **El 5 a 1 no está validado.** De él depende la lectura de la penalización; hay que contrastarlo con el instructor o con datos del banco.
- **Los modelos no están afinados** y ninguna cifra de aquí es una evaluación independiente.

## Supuestos y límites
- Parámetros de partida con esfuerzo parecido para todos, pero no hay garantía de que sean igual de buenos para cada uno.
- Los 15 pliegues comparten filas; las diferencias pareadas pueden salir algo infladas.
- La calibración se midió con las predicciones de los 15 pliegues juntas (cada cliente aparece 3 veces); no se midió por subgrupos.
- La equidad por grupo (`SEX`, edad) no se evaluó en el torneo.
