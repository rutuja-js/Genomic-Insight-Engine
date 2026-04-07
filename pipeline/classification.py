"""
Disease Classification
Trains multiple classifiers on top biomarker genes and evaluates performance.

Models:
- Random Forest     → strong baseline, handles non-linearity, gives feature importance
- Gradient Boosting → often best performer on tabular data
- SVM (RBF kernel)  → effective in high-dimensional spaces

Evaluation:
- Stratified 5-fold cross-validation
- Holdout test set: accuracy, F1 (macro), Cohen's Kappa
- Per-class ROC curves (one-vs-rest)
- Confusion matrix
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


class DiseaseClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=random_state,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=random_state,
            ),
            "SVM_RBF": SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=random_state,
            ),
        }
        self.best_model_ = None
        self.best_model_name_ = None
        self.cv_results_ = {}
        self.test_results_ = {}
        self.roc_data_ = {}

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def cross_validate_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        print("\n=== CLASSIFICATION: CROSS-VALIDATION ===")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "f1_macro", "roc_auc_ovr_weighted"]
        best_score = -1

        for name, model in self.models.items():
            results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            mean_acc  = results["test_accuracy"].mean()
            mean_f1   = results["test_f1_macro"].mean()
            mean_auc  = results["test_roc_auc_ovr_weighted"].mean()

            self.cv_results_[name] = {
                "accuracy_mean": mean_acc, "accuracy_std": results["test_accuracy"].std(),
                "f1_mean": mean_f1,        "f1_std": results["test_f1_macro"].std(),
                "auc_mean": mean_auc,      "auc_std": results["test_roc_auc_ovr_weighted"].std(),
            }
            print(f"[CV] {name:20s} | Acc: {mean_acc:.4f} | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f}")

            if mean_f1 > best_score:
                best_score = mean_f1
                self.best_model_name_ = name
                self.best_model_ = model

        print(f"\n[CV] Best model: {self.best_model_name_} (F1={best_score:.4f})")
        return self.cv_results_

    # ------------------------------------------------------------------
    # Final fit + holdout test evaluation
    # ------------------------------------------------------------------
    def evaluate_on_test(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame,  y_test: pd.Series,
        class_names: list,
    ) -> dict:
        print("\n=== HOLDOUT TEST EVALUATION ===")

        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            acc   = accuracy_score(y_test, y_pred)
            f1    = f1_score(y_test, y_pred, average="macro")
            kappa = cohen_kappa_score(y_test, y_pred)

            results[name] = {
                "accuracy": acc, "f1_macro": f1, "kappa": kappa,
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(
                    y_test, y_pred, target_names=class_names, output_dict=True
                ),
                "y_prob": y_prob,
            }
            print(f"[Test] {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")

        # ROC curves for best model
        n_classes = len(class_names)
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        best_prob = results[self.best_model_name_]["y_prob"]

        self.roc_data_ = {}
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], best_prob[:, i])
            self.roc_data_[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

        self.test_results_ = results
        return results

    # ------------------------------------------------------------------
    # Print best model summary
    # ------------------------------------------------------------------
    def print_best_report(self, class_names: list):
        report = self.test_results_[self.best_model_name_]["classification_report"]
        print(f"\n=== BEST MODEL: {self.best_model_name_} ===")
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 55)
        for cls in class_names:
            m = report[cls]
            print(f"{cls:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {int(m['support']):>10}")
