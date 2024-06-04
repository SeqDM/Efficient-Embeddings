# Source: https://github.com/EleutherAI/pythia
MODEL_TO_LR = {
    'EleutherAI/pythia-14m':  1.0e-3,
    'EleutherAI/pythia-31m':  1.0e-3,
    'EleutherAI/pythia-70m':  1.0e-3,
    'EleutherAI/pythia-160m': 6.0e-4,
    'EleutherAI/pythia-410m': 3.0e-4,
    'EleutherAI/pythia-1b':   3.0e-4,
    'EleutherAI/pythia-1.4b': 2.0e-4,
    'EleutherAI/pythia-2.8b': 1.6e-4,
    'EleutherAI/pythia-6.9b': 1.2e-4,
    'EleutherAI/pythia-12b':  1.2e-4
}

ALL_PYTHIA_MODELS = set(MODEL_TO_LR)

def get_original_lr(model_path):
    return MODEL_TO_LR[model_path]

