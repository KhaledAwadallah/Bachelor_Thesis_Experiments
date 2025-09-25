from rf_train import run_rf_experiment
from bfh_train import run_bfh_experiment
from ftfh_train import run_ftfh_experiment
from maml_train import run_maml_experiment
from data_processing import load_and_preprocess_data
import config
import pickle

if __name__ == "__main__":
    # 1. load and process data
    all_features, muv_matrix = load_and_preprocess_data(config.DATA_PATH)

    # 2. Run experiments
    print("\n\n*************************************************************************************************************\n\n")
    print("--- Starting Random Forest Experiment ---")
    rf_results = run_rf_experiment(all_features, muv_matrix, config.SEEDS, config.RF_TASKS)
    print("\n\n*************************************************************************************************************\n\n")
    print("--- Starting Baseline Frequent Hitters Experiment ---")
    bfh_results = run_bfh_experiment(all_features, muv_matrix, config.SEEDS)
    print("\n\n*************************************************************************************************************\n\n")
    print("--- Starting Fine-tuned Frequent Hitters Experiment ---")
    ftfh_results = run_ftfh_experiment(all_features, muv_matrix, config.SEEDS)
    print("\n\n*************************************************************************************************************\n\n")
    print("--- Starting MAML Experiment ---")
    maml_results = run_maml_experiment(all_features, muv_matrix, config.SEEDS)

    # 3. Store the results
    with open("results.pkl", "wb") as f:
        pickle.dump({"rf_results": rf_results, "bfh_results": bfh_results, "ftfh_results": ftfh_results, "maml_results": maml_results}, f)