import os

# DIRECTORIES ##################################################
PROJ_DIR=os.path.dirname(os.path.realpath(__file__))+'/'
RESULTS_FOLDER = "results/"
DATA_FOLDER = "data/"
FIGURES_FOLDER = "figures/"
CONFIGS_FOLDER = "simulation/configs/"
EXP_DATA_FOLDER = '../exp_data/'

#LOGGING
PRINT_EPOCH = 5

#DATA
MAX_Y_POS = 7
MOVEMENT_SPEED_THRESHOLD = 6.0
# LOSS_THRESHOLD = 0.20
# MAX_TRAINING_TRIALS = 500
# MIN_TRAINING_TRIALS = 50

#RANDOM SEEDS
SEEDS1 = range(1000020,1000030)
SEEDS2 = range(1000050,1000060)

#TRAINING
BATCH_SIZE = 64

#DATA PROCESSING
pca_dims = 10

BIN_SIZE = .01  # sec
n_components = 10 
n_targets = 8
seed_idx_ex = 0
trial_ex = 1

WINDOW_prep = (-.40, .05)  # sec
WINDOW_exec = (-.05, .40)  # sec
WINDOW_prep_exec = (-.40, .40)  # sec
