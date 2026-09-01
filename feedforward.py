"""
Implementacion de una tecnica de aprendizaje sin frameworks. MODULO 2
Mario Alberto Perez Barrera A01799928

Red neuronal feedforward (1 capa oculta) para clasificacion binaria, entrenada con backpropagation manual.
sklearn.datasets se usa UNICAMENTE para generar y dividir los datos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

class FeedforwardNN:
    def __init__(self, n_input, n_hidden, n_output=1,
                 learning_rate=0.02, l2_lambda=0.0, random_state=None):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = learning_rate
        self.l2_lambda = l2_lambda

        rng = np.random.default_rng(random_state)

        # Pesos: distribucion normal estandar
        # Sesgos: valores unitarios.
        self.W1 = rng.standard_normal((n_input, n_hidden))
        self.b1 = np.ones((1, n_hidden))
        self.W2 = rng.standard_normal((n_hidden, n_output))
        self.b2 = np.ones((1, n_output))

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500) 
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2, cache

    def _compute_loss(self, y_true, y_pred):
        n = y_true.shape[0]
        eps = 1e-10  # evita log(0)
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        data_loss = -np.mean(
            y_true * np.log(y_pred_c) + (1 - y_true) * np.log(1 - y_pred_c)
        )
        if self.l2_lambda > 0:
            l2_term = (self.l2_lambda / (2 * n)) * (
                np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
            )
            data_loss += l2_term
        return data_loss

    def _backward(self, X, y_true, cache):
        n = X.shape[0]
        a1, a2 = cache["a1"], cache["a2"]

        # Capa de salida: delta2 = y_hat - y
        delta2 = a2 - y_true
        dW2 = (a1.T @ delta2) / n
        db2 = np.sum(delta2, axis=0, keepdims=True) / n

        # Backprop a la capa oculta
        delta1 = (delta2 @ self.W2.T) * (a1 * (1 - a1))
        dW1 = (X.T @ delta1) / n
        db1 = np.sum(delta1, axis=0, keepdims=True) / n

        if self.l2_lambda > 0:
            dW2 += (self.l2_lambda / n) * self.W2
            dW1 += (self.l2_lambda / n) * self.W1

        # Descenso de gradiente
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X_train, y_train, X_val, y_val, epochs=100,
            early_stopping_patience=None, verbose_every=10):
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            y_pred_train, cache = self._forward(X_train)
            train_loss = self._compute_loss(y_train, y_pred_train)
            self._backward(X_train, y_train, cache)

            y_pred_val, _ = self._forward(X_val)
            val_loss = self._compute_loss(y_val, y_pred_val)

            train_acc = np.mean((y_pred_train >= 0.5).astype(int) == y_train)
            val_acc = np.mean((y_pred_val >= 0.5).astype(int) == y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if verbose_every and epoch % verbose_every == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

            if early_stopping_patience is not None:
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping en epoch {epoch} "
                              f"(sin mejora en {early_stopping_patience} epocas).")
                        break

    def predict_proba(self, X):
        y_pred, _ = self._forward(X)
        return y_pred

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).flatten()

"""
 Metricas de evaluacion: confusion matrix, accuracy, precision, recall, f1-score

"""

def confusion_matrix(y_true, y_pred, n_classes=2):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def classification_report(y_true, y_pred, n_classes=2):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    y_true = np.asarray(y_true)

    report = {}
    total_support = len(y_true)
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for c in range(n_classes):
        tp = cm[c][c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support),
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support

    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["macro avg"] = {
        "precision": macro_p / n_classes,
        "recall": macro_r / n_classes,
        "f1-score": macro_f1 / n_classes,
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": weighted_p / total_support,
        "recall": weighted_r / total_support,
        "f1-score": weighted_f1 / total_support,
        "support": total_support,
    }
    return report


def print_classification_report(report, n_classes=2):
    print(f"{'':>12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for c in range(n_classes):
        r = report[c]
        print(f"{c:>12}{r['precision']:>10.3f}{r['recall']:>10.3f}"
              f"{r['f1-score']:>10.3f}{r['support']:>10}")
    print()
    print(f"{'accuracy':>12}{'':>10}{'':>10}{report['accuracy']:>10.3f}"
          f"{report['weighted avg']['support']:>10}")
    for label in ("macro avg", "weighted avg"):
        r = report[label]
        print(f"{label:>12}{r['precision']:>10.3f}{r['recall']:>10.3f}"
              f"{r['f1-score']:>10.3f}{r['support']:>10}")


"""
======  DATASET  ======
"""
def load_dataset():
    X, y = make_classification(
        n_samples=7000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=RANDOM_STATE,
    )
    return X, y


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, out_path, title="Matriz de confusion - Test set"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("Cargando dataset...")
    X, y = load_dataset()

    # 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | "
          f"Test: {X_test.shape[0]}")

    print("\nEntrenando...")
    nn = FeedforwardNN(
        n_input=X_train.shape[1],
        n_hidden=4,
        n_output=1,
        learning_rate=0.02,
        l2_lambda=0.0,
        random_state=RANDOM_STATE,
    )
    nn.fit(X_train, y_train, X_val, y_val, epochs=100, verbose_every=10)

    print("\nEvaluando en el conjunto de prueba...")
    y_pred_test = nn.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test, n_classes=2)
    report = classification_report(y_test, y_pred_test, n_classes=2)

    print(f"\n=== Accuracy en test: {acc:.4f} ===")
    print("\nMatriz de confusion:")
    print(cm)
    print("\nClassification report:")
    print_classification_report(report, n_classes=2)

    print("\nGraficas guardadas en archivos PNG")
    plot_training_curves(nn.history, "training_curves.png")
    plot_confusion_matrix(cm, "confusion_matrix.png")
    print("Listo: training_curves.png, confusion_matrix.png guardadas.")

    # Ejemplo de prediccion individual
    print("\nEjemplo de prediccion individual (en consola)")
    sample = X_test[0].reshape(1, -1)
    proba = nn.predict_proba(sample)[0][0]
    pred_class = int(proba >= 0.5)
    print(f"Muestra: {sample.flatten()}")
    print(f"Clase real: {y_test[0]} | Probabilidad: {proba:.4f} | "
          f"Clase predicha: {pred_class}")


if __name__ == "__main__":
    main()
