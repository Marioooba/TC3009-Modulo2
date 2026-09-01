# Feedforward Neural Network desde cero

Implementación manual (sin frameworks de ML) de una red neuronal feedforward
con una capa oculta, entrenada con backpropagation para un problema de
clasificación binaria.

## Requisitos

```bash
pip install numpy matplotlib scikit-learn
```

> `scikit-learn` se usa **únicamente** para generar el dataset sintético y
> hacer el split train/val/test — el algoritmo (forward prop, backprop,
> pérdida, métricas) está implementado 100% a mano.

## Cómo correrlo

```bash
python3 feedforward_nn.py
```

## Arquitectura

- Entrada: 2 features
- Capa oculta: 4 neuronas, activación sigmoide
- Capa de salida: 1 neurona, activación sigmoide
- Pérdida: entropía cruzada binaria
- Optimización: descenso de gradiente (learning rate = 0.02), con soporte
  opcional de regularización L2

## Dataset

7,000 muestras sintéticas (`make_classification`), divididas en:
- Entrenamiento: 60% (4,200)
- Validación: 20% (1,400)
- Prueba: 20% (1,400)

## Resultados

Después de 100 épocas, el modelo alcanza **97% de accuracy** en el set de
prueba (42 errores de 1,400: 23 falsos positivos, 19 falsos negativos). El
reporte completo con matriz de confusión, métricas por clase y análisis está
en el PDF del repositorio.

## Archivos

- `feedforward_nn.py` — implementación completa (red, métricas, dataset, main)
- `training_curves.png` — curvas de accuracy/loss por época
- `confusion_matrix.png` — matriz de confusión en el set de prueba
