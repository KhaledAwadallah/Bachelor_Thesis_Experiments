# Frequent Hitters in Few-Shot Drug Discovery

**Short project summary**  
This repository contains the code for the bachelor thesis *"Fine-tuned Frequent Hitters in Drug Discovery""* (Khaled Awadallah, 2025). The work investigates whether *a fine-tuned frequent hitters* can compete with state-of-the-art meta-learning methods like MAML. Four models are implemented:

- **Random Forest (RF)** — standard machine learning baseline.  
- **Baseline Frequent Hitter (BFH)** — Naive frequent hitters method that **ignores** the task support set at prediction time.  
- **Fine-Tuned Frequent Hitter (FTFH)** — A frequent hitters model that is **fine-tuned** on each task’s support set.  
- **MAML** — gradient-based meta-learner (evaluated with 5 and 10 inner steps).
---

## Code overview

Files in this project (what each file implements):

- `data/muv.csv`  
  Raw dataset snapshot (MUV subset) used for experiments / preprocessing input.

- `data_processing.py`  
  Preprocessing and dataset preparation: fingerprint/feature computation.

- `config.py`  
  Centralized experiment configuration and default hyperparameters.

- `models.py`  
  contains neural network architectures for the different implemented models

- `bfh_train.py`  
  Training / evaluation script for the Baseline Frequent Hitter method.

- `ftfh_train.py`  
  Training / evaluation script for the Fine-Tuned Frequent Hitter (per-task fine-tuning on support sets).

- `maml_train.py`  
  MAML meta-training and evaluation script.

- `rf_train.py`  
  Script that trains a Random Forest per test task (single-task baseline).

- `evaluation.py`  
  Metric computation (DAUPRC)

- `plots.py`  
  Plotting utilities used to reproduce the thesis figures (mean ± error bars for the reported metrics).

- `main.py`  
  main file that runs the experiments for all four models and save the results.

---

# Notes
- **Hyperparameter selection:** done on validation tasks only, Test tasks are only used for final performance evaluation.  
- **Reporting convention:** for each random seed I average scores across the three test tasks, then compute the reported mean ± standard deviation **across the 10 seeds** (SD across seeds).
- See `config.py` for the exact hyperparameters used.  
- Implementation details and full experimental discussion (results, tables, figures, and supervisor feedback considerations) are documented in the thesis PDF included with the project.