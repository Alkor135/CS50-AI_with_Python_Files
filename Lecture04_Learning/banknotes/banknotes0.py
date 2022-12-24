import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=1)
# model = GaussianNB()

# Чтение данных из файла
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Подлинный" if row[4] == "0" else "Поддельный"
        })

# Разделите данные на группы обучения и тестирования
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]


def fit_model(model):
    # Обучающая модель на тренировочном наборе
    X_training = [row["evidence"] for row in training]
    y_training = [row["label"] for row in training]
    model.fit(X_training, y_training)

    # Делайте прогнозы на тестовом наборе
    X_testing = [row["evidence"] for row in testing]
    y_testing = [row["label"] for row in testing]
    predictions = model.predict(X_testing)

    # Подсчитайте, насколько хорошо мы справились
    correct = 0
    incorrect = 0
    total = 0
    for actual, predicted in zip(y_testing, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    # Вывод результатов
    print(f"\nРезультаты для модели {type(model).__name__}")
    print(f"Корректных: {correct}")
    print(f"Некорректных: {incorrect}")
    print(f"Точность: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    model = Perceptron()
    fit_model(model)

    model = svm.SVC()
    fit_model(model)

    model = KNeighborsClassifier(n_neighbors=1)
    fit_model(model)

    model = GaussianNB()
    fit_model(model)