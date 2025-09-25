import torch

# Data Path
DATA_PATH = "data/muv.csv"

# Model hyperparameter
# Baseline Frequent Hitters (Neural Network)
INPUT_SIZE = 2248
BFH_HS1 = 64
BFH_HS2 = 32
BFH_HS3 = 16
OUTPUT_SIZE = 1
BFH_NUM_EPOCHS = 5
BFH_BATCH_SIZE = 8192
LEARNING_RATE = 0.01

# Fine-tuned Frequent Hitters (Neural Network)
FTFH_HS1 = 128
FTFH_NUM_EPOCHS = 3
FTFH_NUM_EPISODES = 300
FTFH_BATCH_SIZE = 2048
PRE_TRAINING_LR = 0.00001
FINE_TUNING_LR = 0.0001

# MAML (Neural Network)
MAML_HS1 = 128
META_LR = 0.0001
INNER_LR = 0.1
MAML_NUM_EPISODES = 5
TASKS_PER_META_BATCH = 4


# Experiment Settings
K_SHOT = 5
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
VAL_TASKS = [10, 11, 12]
TEST_TASKS = [13, 14, 15]
RF_TASKS = [13, 14, 15]

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

