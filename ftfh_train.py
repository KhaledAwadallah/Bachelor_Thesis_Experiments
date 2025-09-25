import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data_processing import filter_labels, set_seed, get_standardized_features
from evaluation import compute_dauprc_score
from models import FineTunedFHNN, MoleculeDataset
import config
import copy


# --- Fine-tuned Frequent Hitters Experiment ---
def run_ftfh_experiment(features, label_matrix, seeds):
    print("-------------------------------------------------------------------------------------------------------------")

    labels_train = label_matrix[:, :10]
    labels_val = label_matrix[:, 10:13]
    labels_test = label_matrix[:, 13:16]

    labels_train = filter_labels(labels_train)
    labels_val = filter_labels(labels_val)
    labels_test = filter_labels(labels_test)

    features_train = features[labels_train[:, 0]]
    features_val = features[labels_val[:, 0]]
    features_test = features[labels_test[:, 0]]

    # Standardization
    features_train, features_val, features_test = get_standardized_features(features_train, features_val, features_test)

    # Pre-Training
    pre_trained_models = []
    input_size = config.INPUT_SIZE
    ftfh_hs1 = config.FTFH_HS1
    output_size = config.OUTPUT_SIZE
    ftfh_batch_size = config.FTFH_BATCH_SIZE
    pre_training_lr = config.PRE_TRAINING_LR
    fine_tuning_lr = config.FINE_TUNING_LR
    num_epochs = config.FTFH_NUM_EPOCHS
    num_episodes = config.FTFH_NUM_EPISODES
    k_shot = config.K_SHOT

    for seed in seeds:
        set_seed(seed)

        train_dataset = MoleculeDataset(features_train, labels_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=ftfh_batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))

        pre_trained_model = FineTunedFHNN(input_size, ftfh_hs1, output_size)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=pre_training_lr)

        print(f"--- Seed {seed}: Pre-training Frequent Hitters Model --- ")
        for epoch in range(num_epochs):
            pre_trained_model.train()
            for i, (features, labels) in enumerate(train_loader):
                labels = labels.unsqueeze(1)
                outputs = pre_trained_model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss = {loss.item():.4f} ")

        pre_trained_models.append(copy.deepcopy(pre_trained_model))
        print(f"Pre-training for Seed {seed} complete. Model stored.\n ")


    print("\n##############################################################################################################\n")
    print("\n##############################################################################################################\n")

    # Validation
    fh_val_mean_roc_aucs_seeds = []
    fh_val_mean_dauprcs_seeds = []

    for seed_idx, pre_trained_model in enumerate(pre_trained_models):
        current_seed = seeds[seed_idx]
        set_seed(current_seed)

        print(f"\n--- Seed {current_seed}: Validation Frequent Hitters Model --- ")

        val_auc_results = []
        val_dauprc_results = []

        # Process Validation Tasks
        val_tasks = [10, 11, 12]
        for task in val_tasks:
            task_indices = (labels_val[:, 1] == task)
            task_labels = labels_val[task_indices]
            task_features = features_val[task_indices]

            pos_indices = np.where(task_labels[:, 2] == 1)[0]
            neg_indices = np.where(task_labels[:, 2] == 0)[0]
            np.random.seed(current_seed)
            pos_train_indices = np.random.choice(pos_indices, size=min(k_shot, len(pos_indices)), replace=False)
            neg_train_indices = np.random.choice(neg_indices, size=min(k_shot, len(neg_indices)), replace=False)
            ft_indices = np.concatenate([pos_train_indices, neg_train_indices])
            eval_indices = np.array([i for i in range(len(task_labels)) if i not in ft_indices])

            X_ft, y_ft = task_features[ft_indices], task_labels[ft_indices]
            X_eval, y_eval = task_features[eval_indices], task_labels[eval_indices]

            fine_tuned_model = copy.deepcopy(pre_trained_model)
            ft_optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=fine_tuning_lr)
            ft_dataset = MoleculeDataset(X_ft, y_ft)
            ft_loader = DataLoader(dataset=ft_dataset, batch_size=len(ft_dataset), shuffle=True)

            fine_tuned_model.train()
            print(f"\n Task {task}: ")
            for episode in range(num_episodes):
                for features, labels in ft_loader:
                    labels = labels.unsqueeze(1)
                    outputs = fine_tuned_model(features)
                    loss = criterion(outputs, labels)
                    ft_optimizer.zero_grad()
                    loss.backward()
                    ft_optimizer.step()
                    if (episode+1) % 100 == 0:
                        print(f"Episode {episode+1}/{num_episodes}, Loss = {loss.item():.4f} ")

            fine_tuned_model.eval()
            with torch.no_grad():
                eval_dataset = MoleculeDataset(X_eval, y_eval)
                eval_loader = DataLoader(dataset=eval_dataset, batch_size=len(eval_dataset), shuffle=False)
                for features, labels in eval_loader:
                    outputs = fine_tuned_model(features).squeeze()
                    roc_auc = roc_auc_score(labels.numpy(), outputs.numpy()) if len(np.unique(labels.numpy())) > 1 else np.nan
                    mean_dauprc, _, _ = compute_dauprc_score(outputs, labels, torch.zeros_like(labels))
                    val_auc_results.append(roc_auc)
                    val_dauprc_results.append(mean_dauprc)
            print(f"Validation Task {task} - ROC AUC: {roc_auc:.4f}, DAUPRC: {mean_dauprc:.4f}\n ")
            print("-----------------------------------------------------------------------------------------------")

        fh_val_mean_roc_aucs_seeds.append(np.nanmean(val_auc_results))
        fh_val_mean_dauprcs_seeds.append(np.nanmean(val_dauprc_results))
        print(f"\n Seed {current_seed} Average Validation - ROC AUC: {np.nanmean(val_auc_results):.4f}, DAUPRC: {np.nanmean(val_dauprc_results):.4f}\n ")
        print("#####################################################################################################")

    print(f"\n--- Overall Fine-tuned Frequent Hitters Results on Validation Set --- ")
    print(f"Mean ROC AUC Score over all seeds (Validation): {np.nanmean(fh_val_mean_roc_aucs_seeds):.4f}, Standard Deviation: {np.nanstd(fh_val_mean_roc_aucs_seeds):.4f} ")
    print(f"Mean DAUPRC Score over all seeds (Validation): {np.nanmean(fh_val_mean_dauprcs_seeds):.4f}, Standard Deviation: {np.nanstd(fh_val_mean_dauprcs_seeds):.4f} ")

    print("\n##############################################################################################################\n")
    print( "\n##############################################################################################################\n")

    # Testing
    fh_test_mean_roc_aucs_seeds = []
    fh_test_mean_dauprcs_seeds = []

    for seed_idx, pre_trained_model in enumerate(pre_trained_models):
        current_seed = seeds[seed_idx]
        set_seed(current_seed)

        print(f"\n--- Seed {current_seed}: Testing Frequent Hitters Model ---\n ")

        test_auc_results = []
        test_dauprc_results = []

        # Process Test Tasks
        test_tasks = [13, 14, 15]
        for task in test_tasks:
            print(f"Task {task}:")
            task_indices = (labels_test[:, 1] == task)
            task_labels = labels_test[task_indices]
            task_features = features_test[task_indices]

            pos_indices = np.where(task_labels[:, 2] == 1)[0]
            neg_indices = np.where(task_labels[:, 2] == 0)[0]
            np.random.seed(current_seed)
            pos_train_indices = np.random.choice(pos_indices, size=min(5, len(pos_indices)), replace=False)
            neg_train_indices = np.random.choice(neg_indices, size=min(5, len(neg_indices)), replace=False)
            ft_indices = np.concatenate([pos_train_indices, neg_train_indices])
            eval_indices = np.array([i for i in range(len(task_labels)) if i not in ft_indices])

            X_ft, y_ft = task_features[ft_indices], task_labels[ft_indices]
            X_eval, y_eval = task_features[eval_indices], task_labels[eval_indices]

            fine_tuned_model = copy.deepcopy(pre_trained_model)
            ft_optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=fine_tuning_lr)
            ft_dataset = MoleculeDataset(X_ft, y_ft)
            ft_loader = DataLoader(dataset=ft_dataset, batch_size=len(ft_dataset), shuffle=True)

            fine_tuned_model.train()
            for _ in range(num_episodes):
                for features, labels in ft_loader:
                    labels = labels.unsqueeze(1)
                    outputs = fine_tuned_model(features)
                    loss = criterion(outputs, labels)
                    ft_optimizer.zero_grad()
                    loss.backward()
                    ft_optimizer.step()

            fine_tuned_model.eval()
            with torch.no_grad():
                eval_dataset = MoleculeDataset(X_eval, y_eval)
                eval_loader = DataLoader(dataset=eval_dataset, batch_size=len(eval_dataset), shuffle=False)
                for features, labels in eval_loader:
                    outputs = fine_tuned_model(features).squeeze()
                    roc_auc = roc_auc_score(labels.numpy(), outputs.numpy())
                    mean_dauprc, _, _ = compute_dauprc_score(outputs, labels, torch.zeros_like(labels))
                    test_auc_results.append(roc_auc)
                    test_dauprc_results.append(mean_dauprc)
            print(f"ROC AUC: {roc_auc:.4f}, DAUPRC: {mean_dauprc:.4f}")

        fh_test_mean_roc_aucs_seeds.append(np.nanmean(test_auc_results))
        fh_test_mean_dauprcs_seeds.append(np.nanmean(test_dauprc_results))
        print(f"\nSeed {current_seed} Average Test - ROC AUC: {np.nanmean(test_auc_results):.4f}, DAUPRC: {np.nanmean(test_dauprc_results):.4f}\n")

    # Final Results Reporting
    print(f"\n--- Overall Fine-tuned Frequent Hitters Results (FINAL Score ON TEST SET) ---")
    print(
        f"Mean ROC AUC Score over all seeds (Test): {np.nanmean(fh_test_mean_roc_aucs_seeds):.4f}, Standard Deviation: {np.nanstd(fh_test_mean_roc_aucs_seeds):.4f}")
    print(
        f"Mean DAUPRC Score over all seeds (Test): {np.nanmean(fh_test_mean_dauprcs_seeds):.4f}, Standard Deviation: {np.nanstd(fh_test_mean_dauprcs_seeds):.4f}")
    return [fh_test_mean_roc_aucs_seeds, fh_test_mean_dauprcs_seeds]