import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processing import filter_labels, get_standardized_features, set_seed
from models import MoleculeDataset, BaselineFHNN
from evaluation import compute_roc_auc_score, compute_dauprc_score
import config


# --- Frequent Hitters Experiment ---
def run_bfh_experiment(features, label_matrix, seeds):
    print("-------------------------------------------------------------------------------------------------------------")

    mean_roc_auc_seeds, std_roc_auc_seeds = [], []
    mean_dauprc_seeds, std_dauprc_seeds = [], []

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

    for seed in seeds:
        set_seed(seed)
        batch_size = config.BFH_BATCH_SIZE

        # Dataset and Dataloader
        train_dataset = MoleculeDataset(features_train, labels_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

        val_tasks = config.VAL_TASKS
        val_datasets = {}
        val_loaders = {}
        for task in val_tasks:
            val_datasets[task] = MoleculeDataset(features_val[labels_val[:, 1] == task], labels_val[labels_val[:, 1] == task])
        for task, dataset in val_datasets.items():
            val_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

        test_tasks = config.TEST_TASKS
        test_datasets = {}
        test_loaders = {}
        for task in test_tasks:
            test_datasets[task] = MoleculeDataset(features_test[labels_test[:, 1] == task], labels_test[labels_test[:, 1] == task])
        for task, dataset in test_datasets.items():
            test_loaders[task] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

        # Model and hyperparameters
        input_size = features_train.shape[1]
        hs1 = config.BFH_HS1
        hs2 = config.BFH_HS2
        hs3 = config.BFH_HS3
        output_size = 1
        num_epochs = config.BFH_NUM_EPOCHS
        num_steps = len(train_loader)
        learning_rate = config.LEARNING_RATE
        model = BaselineFHNN(input_size, hs1, hs2, hs3, output_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        print(f"\nSeed {seed}:")
        for epoch in range(num_epochs):
            model.train()
            for i, (features, labels) in enumerate(train_loader):
                labels = labels.unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{num_steps}, Loss = {loss}")
            print()

            # Validation
            model.eval()
            with torch.no_grad():
                for task, val_loader in val_loaders.items():
                    val_labels = []
                    val_preds = []
                    for features, labels in val_loader:
                        labels = labels.unsqueeze(1)
                        outputs = model(features)
                        val_preds.extend(outputs.numpy().flatten())
                        val_labels.extend(labels.numpy().flatten())

                    val_labels = np.array(val_labels, dtype=int)
                    val_preds = np.array(val_preds)

                    # ROC AUC Score
                    roc_auc_val = compute_roc_auc_score(val_labels, val_preds)

                    # DAUPRC Score
                    val_labels_tensor = torch.tensor(val_labels)
                    val_preds_tensor = torch.tensor(val_preds)
                    target_ids_tensor_val = torch.zeros_like(val_labels_tensor)
                    mean_dauprc_val, dauprcs_val, target_id_list_val = compute_dauprc_score(
                        val_preds_tensor, val_labels_tensor, target_ids_tensor_val
                    )
                    print(f"Validation Task {task}, ROC AUC Score: {roc_auc_val:.4f}, Mean DAUPRC Score: {mean_dauprc_val:.4f}")
                print()

        # Testing
        print(f"\nResults on the test set:")
        auc_results_fh = []
        dauprc_results_fh = []
        model.eval()
        with torch.no_grad():
            for task, test_loader in test_loaders.items():
                test_preds = []
                test_labels = []
                for features, labels in test_loader:
                    labels = labels.unsqueeze(1)
                    outputs = model(features)
                    test_preds.extend(outputs.numpy().flatten())
                    test_labels.extend(labels.numpy().flatten())

                test_labels = np.array(test_labels, dtype=int)
                test_preds = np.array(test_preds)

                # ROC AUC Score
                roc_auc_test = compute_roc_auc_score(test_labels, test_preds)

                # DAUPRC Score
                test_preds_tensor = torch.tensor(test_preds)
                test_labels_tensor = torch.tensor(test_labels)
                target_ids_tensor_test = torch.zeros_like(test_labels_tensor)
                mean_dauprc_test, dauprcs_test, target_id_list_test = compute_dauprc_score(
                    test_preds_tensor, test_labels_tensor, target_ids_tensor_test
                )

                auc_results_fh.append(roc_auc_test)
                dauprc_results_fh.append(mean_dauprc_test)
                print(f"Test task {task}, ROC AUC Score: {roc_auc_test:.4f}, Mean DAUPRC Score: {mean_dauprc_test:.4f}")

        mean_roc_auc_seeds.append(np.mean(auc_results_fh))
        mean_dauprc_seeds.append(np.nanmean(dauprc_results_fh))

        print(f"\nMean ROC AUC Score across all test tasks: {np.mean(auc_results_fh):.4f}")
        print(f"Mean DAUPRC Score across all test tasks: {np.nanmean(dauprc_results_fh):.4f}")

    print("\n############################################################################################################################\n")
    print(f"Mean ROC AUC Score over all seeds: {np.mean(mean_roc_auc_seeds):.4f},"
          f"Standard Deviation of ROC AUC Score over all seeds: {np.std(mean_roc_auc_seeds):.4f}")
    print(f"Mean DAUPRC Score over all seeds: {np.mean(mean_dauprc_seeds):.4f},"
          f"Standard Deviation of DAUPRC Score over all seeds: {np.std(mean_dauprc_seeds):.4f}")
    return [mean_roc_auc_seeds, mean_dauprc_seeds]