# El objetivo de este experimento

Saber de varias formas, si la red olvida lo que ya ha aprendido. Es decir, si lo sobre escribe
o que ocurre.

Para esto vamos a entrenar el modelo con el conjunto de datos MNIST.
Separandolo en dos fases. En la primera entrena las clases de 0 a 4 y en la 
segunda fase entrena las clases de 5 a 9.

Con los siguientes parametros:

| Parámetro          | Valor                                       |
|---------------------|---------------------------------------------|
| batch_size          | 32                                          |
| learning_rate       | 1e-3                                        |
| loss_fn             | CrossEntropy                                |
| optimizer           | SG                                          |

## Experimentos base
- Entrenar un modelo en dos fases como lo anterior mencionado.

En uno vamos a colocar que en cada fase entrene para 10 clases.
En el otro vamos hacer que entrene para solo 5 clases y en la siguiente para 1o clases.

## ¿Que queremos analizar?

- Queremos ver si el clasificador se descuadra para las clases de la fase anterior.
- Si aún puede clasificar con algún clasificador clases de la fase anterior y la fase realizada.
- Si se superponen las caracteristicas de las clases.