# folder management
import os

HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
DATASETS = os.path.join(DATA, 'datasets')
QUALITY_MONITORING = os.path.join(DATASETS, 'quality_monitoring')
TRANINING_MODE = os.path.join(DATASETS, 'training_mode')
PROCESS_PARAMETERS = os.path.join(DATASETS, 'process_parameters')

PREDICTION_MODELS = os.path.join(DATA, 'prediction_models')

WEIGHTS = os.path.join(DATA, 'weights')