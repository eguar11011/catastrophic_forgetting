# El objetivo de este experimento

El objetivo de este experimento es investigar la capacidad de una red neuronal para retener y generalizar el conocimiento adquirido durante el entrenamiento en dos fases. Se plantea la pregunta de si la red olvida lo aprendido en la primera fase cuando se entrena en la segunda, y cómo esto afecta la capacidad del modelo para clasificar las clases de ambos conjuntos de datos.

El experimento utiliza el conjunto de datos MNIST y se divide en dos fases. En la primera fase, se entrena el modelo con las clases de dígitos del 0 al 4, y en la segunda fase, se entrena con las clases de dígitos del 5 al 9. Se realizan dos experimentos base con diferentes enfoques:

## Experimento 1: Entrenamiento para 5 clases y luego para 10 clases
- En este experimento, se entrena el modelo inicialmente para las primeras 5 clases en la primera fase y luego se amplía el entrenamiento para las 10 clases en la segunda fase.
- Se busca entender cómo el modelo maneja la transición de un conjunto de clases a otro, y si puede clasificar de manera efectiva tanto las clases de la fase anterior como las nuevas.



Se proponen varias métricas y análisis para evaluar el rendimiento, que incluyen:

- La visualización de características, 
- La comparación de predicciones en ambas fases.
- El análisis detallado de los pesos de los dos modelos resultantes.
- Analizar el clasificador.

 El experimento busca identificar qué conocimientos son considerados importantes por el modelo y si existe alguna pérdida significativa de información entre las fases.
