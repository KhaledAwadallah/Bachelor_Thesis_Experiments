import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from data_processing import filter_labels
from evaluation import compute_roc_auc_score, compute_dauprc_score
from models import get_random_forest_classifier


# --- Random Forest Experiment ---
def run_rf_experiment(features, label_matrix, seeds, tasks):
    auc_results = {seed: [] for seed in seeds}
    dauprc_results = {seed: [] for seed in seeds}
    label_matrix = label_matrix[:, 13:16]
    filtered_matrix = filter_labels(label_matrix)

    for seed in seeds:
        np.random.seed(seed)
        for task in tasks:
            data = filtered_matrix[filtered_matrix[:, 1] == task]
            X = features[data[:, 0]]
            y = data[:, 2]
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            pos_train_indices = np.random.choice(pos_indices, size=5, replace=False)
            neg_train_indices = np.random.choice(neg_indices, size=5, replace=False)
            train_indices = np.concatenate([pos_train_indices, neg_train_indices])
            test_indices = [i for i in range(len(y)) if i not in train_indices]
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            clf = get_random_forest_classifier(seed)
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            predictions_tensor = torch.tensor(y_pred_proba)
            labels_tensor = torch.tensor(y_test)
            target_ids_tensor = torch.zeros_like(labels_tensor)

            auc_score = compute_roc_auc_score(y_test, y_pred_proba)
            mean_dauprc, dauprcs, target_id_list = compute_dauprc_score(
                predictions_tensor,
                labels_tensor,
                target_ids_tensor
            )

            auc_results[seed].append(auc_score)
            dauprc_results[seed].append(mean_dauprc)

    mean_auc_seeds, std_auc_seeds = [], []
    mean_dauprc_seeds, std_dauprc_seeds = [], []
    for seed in seeds:
        auc_scores = auc_results[seed]
        mean_auc = np.mean(auc_scores)
        mean_auc_seeds.append(mean_auc)
        print(f"Seed {seed}:")
        print(f"Mean roc_auc_score = {mean_auc:.4f}")

        dauprcs = dauprc_results[seed]
        mean_dauprc = np.nanmean(dauprcs)
        mean_dauprc_seeds.append(mean_dauprc)
        print(f"Mean dauprc_score = {mean_dauprc:.4f}\n")

    print(f"\nMean AUC Score over all seeds: {np.mean(mean_auc_seeds):.4f},"
          f"Standard Deviation of AUC Score over all seeds: {np.std(mean_auc_seeds):.4f}")
    print(f"Mean DAUPRC Score over all seeds: {np.nanmean(mean_dauprc_seeds):.4f},"
          f"Standard Deviation of DAUPRC Score over all seeds: {np.nanstd(mean_dauprc_seeds):.4f}")

    return [mean_auc_seeds, mean_dauprc_seeds]