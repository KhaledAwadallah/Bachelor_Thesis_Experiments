import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def plot_performance(df):
    sns.set(style="whitegrid", font_scale=1.2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. ROC AUC Comparison (averaged)
    sns.barplot(data=df, x="model", y="roc_auc", errorbar="sd", capsize=0.2, ax=axes[0, 0])
    axes[0, 0].set_title("Average ROC AUC Score Across Models")
    axes[0, 0].set_ylabel("ROC AUC")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # 2. DAUPRC Comparison (averaged)
    sns.barplot(data=df, x="model", y="dauprc", errorbar="sd", capsize=0.2, ax=axes[0, 1])
    axes[0, 1].set_title("Average DAUPC Score Across Models")
    axes[0, 1].set_ylabel("DAUPRC")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # 3. ROC AUC Per Seed
    sns.lineplot(data=df, x="seed", y="roc_auc", hue="model", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("ROC AUC Score Per Seed")
    axes[1, 0].set_ylabel("ROC AUC")
    axes[1, 0].set_xlabel("Seed")
    axes[1, 0].set_xticks(sorted(df["seed"].unique()))

    # 4. DAUPRC Per Seed
    sns.lineplot(data=df, x="seed", y="dauprc", hue="model", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title("DAUPRC Score Per Seed")
    axes[1, 1].set_ylabel("DAUPRC")
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_xticks(sorted(df["seed"].unique()))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("results.pkl", "rb") as f:
        data = pickle.load(f)

        results = []
        rf_results = data["rf_results"]
        bfh_results = data["bfh_results"]
        ftfh_results = data["ftfh_results"]
        maml_results = data["maml_results"]

        rf_test_mean_roc_aucs_seeds, rf_test_mean_dauprcs_seeds = rf_results[0], rf_results[1]
        bfh_test_mean_roc_aucs_seeds, bfh_test_mean_dauprcs_seeds = bfh_results[0], bfh_results[1]
        ftfh_test_mean_roc_aucs_seeds, ftfh_test_mean_dauprcs_seeds = ftfh_results[0], ftfh_results[1]
        maml_test_mean_roc_aucs_seeds, maml_test_mean_dauprcs_seeds = maml_results[0], maml_results[1]

        for seed, (roc, dau) in enumerate(zip(rf_test_mean_roc_aucs_seeds, rf_test_mean_dauprcs_seeds)):
            results.append({"model": "Random Forest", "seed": seed, "roc_auc": roc, "dauprc": dau})

        for seed, (roc, dau) in enumerate(zip(bfh_test_mean_roc_aucs_seeds, bfh_test_mean_dauprcs_seeds)):
            results.append({"model": "Baseline Frequent Hitters", "seed": seed, "roc_auc": roc, "dauprc": dau})

        for seed, (roc, dau) in enumerate(zip(ftfh_test_mean_roc_aucs_seeds, ftfh_test_mean_dauprcs_seeds)):
            results.append({"model": "Fine-tuned Frequent Hitters", "seed": seed, "roc_auc": roc, "dauprc": dau})

        for seed, (roc, dau) in enumerate(zip(maml_test_mean_roc_aucs_seeds, maml_test_mean_dauprcs_seeds)):
            results.append({"model": "MAML (5 gradient steps)", "seed": seed, "roc_auc": roc, "dauprc": dau})

        results_df = pd.DataFrame(results)
        plot_performance(results_df)



