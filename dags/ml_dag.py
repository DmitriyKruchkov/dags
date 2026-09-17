"""
### ML Training Pipeline

Учебный DAG, который эмулирует полноценный ML-пайплайн:

1. `generate_data`      — генерирует синтетический датасет (классификация)
2. `preprocess`         — делит на train/test, масштабирует признаки
3. `train_logreg`       \
4. `train_random_forest` } — обучают разные модели ПАРАЛЛЕЛЬНО
5. `train_gboost`       /
6. `evaluate_and_select`— сравнивает метрики, выбирает лучшую модель
7. `save_report`        — сохраняет итоговый отчёт

Все данные и модели передаются между тасками через XCom
(для учебных целей; в проде для больших объёмов лучше
использовать внешнее хранилище — S3/GCS/PVC — и передавать пути).
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta

from airflow.sdk import DAG, task

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_training_pipeline",
    description="Синтетический ML-пайплайн: генерация данных -> обучение нескольких моделей параллельно -> выбор лучшей",
    default_args=default_args,
    schedule=None,  # запускается только вручную/по триггеру
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["example", "ml", "sklearn"],
) as dag:

    dag.doc_md = __doc__

    @task
    def generate_data(n_samples: int = 5000, n_features: int = 20) -> dict:
        """Генерирует синтетический датасет для бинарной классификации."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=12,
            n_redundant=4,
            n_classes=2,
            random_state=42,
        )
        return {"X": X.tolist(), "y": y.tolist()}

    @task
    def preprocess(data: dict) -> dict:
        """Train/test split + масштабирование признаков."""
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.array(data["X"])
        y = np.array(data["y"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return {
            "X_train": X_train.tolist(),
            "X_test": X_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
        }

    @task
    def train_logreg(split: dict) -> dict:
        """Обучает логистическую регрессию."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        X_train, y_train = np.array(split["X_train"]), np.array(split["y_train"])
        X_test, y_test = np.array(split["X_test"]), np.array(split["y_test"])

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model_name": "logistic_regression",
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

    @task
    def train_random_forest(split: dict) -> dict:
        """Обучает Random Forest."""
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score

        X_train, y_train = np.array(split["X_train"]), np.array(split["y_train"])
        X_test, y_test = np.array(split["X_test"]), np.array(split["y_test"])

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model_name": "random_forest",
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

    @task
    def train_gboost(split: dict) -> dict:
        """Обучает Gradient Boosting."""
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score

        X_train, y_train = np.array(split["X_train"]), np.array(split["y_train"])
        X_test, y_test = np.array(split["X_test"]), np.array(split["y_test"])

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model_name": "gradient_boosting",
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

    @task
    def evaluate_and_select(results: list[dict]) -> dict:
        """Сравнивает модели по F1 и выбирает лучшую."""
        best = max(results, key=lambda r: r["f1"])
        return {
            "best_model": best["model_name"],
            "best_f1": best["f1"],
            "all_results": results,
        }

    @task
    def save_report(summary: dict) -> None:
        """Печатает итоговый отчёт (в проде — запись в БД/S3/уведомление)."""
        print("=" * 50)
        print("ML PIPELINE REPORT")
        print("=" * 50)
        for r in summary["all_results"]:
            print(f"  {r['model_name']:<20} accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}")
        print("-" * 50)
        print(f"BEST MODEL: {summary['best_model']} (f1={summary['best_f1']:.4f})")
        print("=" * 50)

    # --- Граф зависимостей ---
    raw_data = generate_data()
    split_data = preprocess(raw_data)

    results = [
        train_logreg(split_data),
        train_random_forest(split_data),
        train_gboost(split_data),
    ]

    summary = evaluate_and_select(results)
    save_report(summary)    