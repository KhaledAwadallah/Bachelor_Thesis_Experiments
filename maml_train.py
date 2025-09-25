import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from data_processing import filter_labels, set_seed, get_standardized_features
from evaluation import compute_dauprc_score
from models import MAMLNN
import config
import copy


def maml_meta_training(meta_model, features, labels, num_episodes, tasks_per_meta_batch, k_shot, inner_lr, meta_lr, num_adaptation_steps=5):
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    criterion = nn.BCELoss()
    train_tasks = np.unique(labels[:, 1])
    print("--- Starting MAML Meta-Training ---")

    for episode in range(num_episodes):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        sampled_tasks = np.random.choice(train_tasks, size=min(tasks_per_meta_batch, len(train_tasks)), replace=False)

        for task_id in sampled_tasks:
            task_indices = np.where(labels[:, 1] == task_id)[0]
            task_labels = labels[task_indices]
            task_features = features[task_indices]

            pos_indices = np.where(task_labels[:, 2] == 1)[0]
            neg_indices = np.where(task_labels[:, 2] == 0)[0]

            support_pos_indices = np.random.choice(pos_indices, size=k_shot, replace=False)
            support_neg_indices = np.random.choice(neg_indices, size=k_shot, replace=False)
            support_indices = np.concatenate([support_pos_indices, support_neg_indices])
            query_indices = np.array([i for i in range(len(task_labels)) if i not in support_indices])

            X_support_np, y_support_np = task_features[support_indices], task_labels[support_indices]
            X_query_np, y_query_np = task_features[query_indices], task_labels[query_indices]

            X_support = torch.tensor(X_support_np, dtype=torch.float32)
            y_support = torch.tensor(y_support_np[:, 2], dtype=torch.float32).unsqueeze(1)
            X_query = torch.tensor(X_query_np, dtype=torch.float32)
            y_query = torch.tensor(y_query_np[:, 2], dtype=torch.float32).unsqueeze(1)

            # Inner Loop: Fast Adaptation
            fast_weights = {name: param for name, param in meta_model.named_parameters()}

            for step in range(num_adaptation_steps):
                support_outputs = meta_model.functional_forward(X_support, fast_weights)
                inner_loss = criterion(support_outputs, y_support)

                grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)

                fast_weights = {
                    name: param - inner_loss * grad
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }

            # Outer Loop: Evaluate with adapted weights
            query_outputs = meta_model.functional_forward(X_query, fast_weights)
            query_loss = criterion(query_outputs, y_query)

            meta_loss += query_loss

        # Meta-update
        meta_loss.backward()
        meta_optimizer.step()

        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Meta Loss = {meta_loss.item():.4f}")

    return meta_model


def run_maml_experiment(features, label_matrix, seeds):
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

    input_size = config.INPUT_SIZE
    maml_hs1 = config.MAML_HS1
    output_size = config.OUTPUT_SIZE
    num_episodes = config.MAML_NUM_EPISODES
    tasks_per_meta_batch = config.TASKS_PER_META_BATCH
    k_shot = config.K_SHOT
    inner_lr = config.INNER_LR
    meta_lr = config.META_LR

    # --- Training ---
    meta_trained_models = []
    for seed in seeds:
        set_seed(seed)
        # Initialize a new model for each seed
        meta_model = MAMLNN(input_size, maml_hs1, output_size)

        # Perform MAML Meta-Training
        trained_model = maml_meta_training(
            meta_model=meta_model,
            features=features_train,
            labels=labels_train,
            num_episodes=num_episodes,
            tasks_per_meta_batch=tasks_per_meta_batch,
            k_shot=k_shot,
            inner_lr=inner_lr,
            meta_lr=meta_lr
        )

        meta_trained_models.append(copy.deepcopy(trained_model))
        print(f"Meta-training for Seed {seed} complete. Model stored.\n ")

    print("\n##############################################################################################################\n")
    print("\n##############################################################################################################\n")

    # --- Validation ---
    maml_val_mean_roc_aucs_seeds = []
    maml_val_mean_dauprcs_seeds = []
    criterion = nn.BCELoss()

    print("\n--- Starting Validation with MAML-trained Models ---")
    for seed_idx, meta_trained_model in enumerate(meta_trained_models):
        current_seed = seeds[seed_idx]
        set_seed(current_seed)

        print(f"\n--- Seed {current_seed}: Validation MAML Model --- ")
        val_auc_results = []
        val_dauprc_results = []
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

            X_support = torch.tensor(X_ft, dtype=torch.float32)
            y_support = torch.tensor(y_ft[:, 2], dtype=torch.float32).unsqueeze(1)
            X_query = torch.tensor(X_eval, dtype=torch.float32)
            y_query = torch.tensor(y_eval[:, 2], dtype=torch.float32)

            adapted_model = copy.deepcopy(meta_trained_model)
            adapted_model.train()

            # Fast adaptation with 5 gradient steps
            num_adaptation_steps = 5
            for step in range(num_adaptation_steps):
                support_outputs = adapted_model(X_support)
                adaptation_loss = criterion(support_outputs, y_support)

                # Compute gradients
                grads = torch.autograd.grad(adaptation_loss, adapted_model.parameters(), create_graph=False)

                # Manually update parameters
                with torch.no_grad():
                    for param, grad in zip(adapted_model.parameters(), grads):
                        param.data = param.data - inner_lr * grad

            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                query_outputs = adapted_model(X_query).squeeze()
                roc_auc = roc_auc_score(y_query.numpy(), query_outputs.numpy())
                mean_dauprc, _, _ = compute_dauprc_score(query_outputs, y_query, torch.zeros_like(y_query))
                val_auc_results.append(roc_auc)
                val_dauprc_results.append(mean_dauprc)

            print(f"Validation Task {task} - ROC AUC: {roc_auc:.4f}, DAUPRC: {mean_dauprc:.4f} ")

        maml_val_mean_roc_aucs_seeds.append(np.nanmean(val_auc_results))
        maml_val_mean_dauprcs_seeds.append(np.nanmean(val_dauprc_results))
        print(
            f"\nSeed {current_seed}: Average Validation - ROC AUC: {np.nanmean(val_auc_results):.4f}, DAUPRC: {np.nanmean(val_dauprc_results):.4f}\n ")

    print(f"\n--- Overall MAML Results on Validation Set --- ")
    print(
        f"Mean ROC AUC Score over all seeds (Validation): {np.nanmean(maml_val_mean_roc_aucs_seeds):.4f}, Standard Deviation: {np.nanstd(maml_val_mean_roc_aucs_seeds):.4f} ")
    print(
        f"Mean DAUPRC Score over all seeds (Validation): {np.nanmean(maml_val_mean_dauprcs_seeds):.4f}, Standard Deviation: {np.nanstd(maml_val_mean_dauprcs_seeds):.4f} ")

    print("\n##############################################################################################################\n")
    print("\n##############################################################################################################\n")

    # --- Testing ---
    maml_test_mean_roc_aucs_seeds = []
    maml_test_mean_dauprcs_seeds = []
    criterion = nn.BCELoss()

    print("\n--- Starting Testing with MAML-trained Models ---")
    for seed_idx, meta_trained_model in enumerate(meta_trained_models):
        current_seed = seeds[seed_idx]
        set_seed(current_seed)

        print(f"\n--- Seed {current_seed}: Testing MAML Model --- ")
        test_auc_results = []
        test_dauprc_results = []
        test_tasks = [13, 14, 15]
        for task in test_tasks:
            task_indices = (labels_test[:, 1] == task)
            task_labels = labels_test[task_indices]
            task_features = features_test[task_indices]

            pos_indices = np.where(task_labels[:, 2] == 1)[0]
            neg_indices = np.where(task_labels[:, 2] == 0)[0]

            np.random.seed(current_seed)
            pos_train_indices = np.random.choice(pos_indices, size=min(k_shot, len(pos_indices)), replace=False)
            neg_train_indices = np.random.choice(neg_indices, size=min(k_shot, len(neg_indices)), replace=False)
            ft_indices = np.concatenate([pos_train_indices, neg_train_indices])
            eval_indices = np.array([i for i in range(len(task_labels)) if i not in ft_indices])

            X_ft, y_ft = task_features[ft_indices], task_labels[ft_indices]
            X_eval, y_eval = task_features[eval_indices], task_labels[eval_indices]

            X_support = torch.tensor(X_ft, dtype=torch.float32)
            y_support = torch.tensor(y_ft[:, 2], dtype=torch.float32).unsqueeze(1)
            X_query = torch.tensor(X_eval, dtype=torch.float32)
            y_query = torch.tensor(y_eval[:, 2], dtype=torch.float32)

            adapted_model = copy.deepcopy(meta_trained_model)
            adapted_model.train()

            # Fast adaptation with 5 gradient steps
            num_adaptation_steps = 5
            for step in range(num_adaptation_steps):
                support_outputs = adapted_model(X_support)
                adaptation_loss = criterion(support_outputs, y_support)

                # Compute_gradients
                grads = torch.autograd.grad(adaptation_loss, adapted_model.parameters(), create_graph=False)

                # Manually update parameters
                with torch.no_grad():
                    for param, grad in zip(adapted_model.parameters(), grads):
                        param.data = param.data - inner_lr * grad

            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                query_outputs = adapted_model(X_query).squeeze()
                roc_auc = roc_auc_score(y_query.numpy(), query_outputs.numpy())
                mean_dauprc, _, _ = compute_dauprc_score(query_outputs, y_query, torch.zeros_like(y_query))
                test_auc_results.append(roc_auc)
                test_dauprc_results.append(mean_dauprc)

            print(f"Test Task {task} - ROC AUC: {roc_auc:.4f}, DAUPRC: {mean_dauprc:.4f} ")

        maml_test_mean_roc_aucs_seeds.append(np.nanmean(test_auc_results))
        maml_test_mean_dauprcs_seeds.append(np.nanmean(test_dauprc_results))
        print(
            f"\nSeed {current_seed}: Average Test Score - ROC AUC: {np.nanmean(test_auc_results):.4f}, DAUPRC: {np.nanmean(test_dauprc_results):.4f}\n ")

    print(f"\n--- Overall MAML Results on Test set ---")
    print(
        f"Mean ROC AUC Score over all seeds (Test): {np.nanmean(maml_test_mean_roc_aucs_seeds):.4f}, Standard Deviation: {np.nanstd(maml_test_mean_roc_aucs_seeds):.4f} ")
    print(
        f"Mean DAUPRC Score over all seeds (Test): {np.nanmean(maml_test_mean_dauprcs_seeds):.4f}, Standard Deviation: {np.nanstd(maml_test_mean_dauprcs_seeds):.4f} ")

    return [maml_test_mean_roc_aucs_seeds, maml_test_mean_dauprcs_seeds]