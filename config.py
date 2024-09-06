import os

REPO_PATH = os.getcwd().split('IR-LLM-for-Effort-Estimation')[0] + 'IR-LLM-for-Effort-Estimation'
DATA_PATH = REPO_PATH + '/UQV100-and-180-selected-TREC-topics'
UQV100_DATA_PATH = DATA_PATH + '/3180694'

LOGS_PATH = REPO_PATH + '/logs'
GRAPHICS_PATH = LOGS_PATH + '/graphics'
BUCKET_GRAPHICS_PATH = GRAPHICS_PATH +'/bucketed'

AGGREGATED_ANNOTATIONS = LOGS_PATH + '/aggregated_annotations'
BUCKET_PREDS = AGGREGATED_ANNOTATIONS +'/bucketing'
LLMS_PREDICTIONS = LOGS_PATH + '/llms_predictions'
LLMS_PREDICTIONS_DIST = LLMS_PREDICTIONS + '/pred_distributions'

FAST_DAWID_SKENE_REPO = REPO_PATH + '/Fast-Dawid-Skene'
FAST_DAWID_SKENE_DATA = FAST_DAWID_SKENE_REPO + '/data'