# Mario Alberto Perez Barrera A01799928
# Feedforward Neural Network desde cero

Implementación manual de una red neuronal feedforward
con una capa oculta, entrenada con backpropagation.

## Cómo correrlo

```bash
python3 feedforward.py
```

## Arquitectura

- Entrada: 2 features
- Capa oculta: 4 neuronas, activación sigmoide
- Capa de salida: 1 neurona, activación sigmoide
- Pérdida: entropía cruzada binaria
- Optimización: descenso de gradiente (learning rate = 0.02)

## Dataset

7,000 muestras sintéticas (`make_classification`), divididas en:
- Entrenamiento: 60% (4,200)
- Validación: 20% (1,400)
- Prueba: 20% (1,400)

## Archivos

- `feedforward_nn.py` — implementación completa (red, métricas, dataset, main)
- `training_curves.png` — curvas de accuracy/loss por época
- `confusion_matrix.png` — matriz de confusión en el set de prueba
