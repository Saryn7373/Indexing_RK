import numpy as np
from gensim.models import Word2Vec

from utils.paths import ROOT_DIR

MODELS_DIR = ROOT_DIR / "task08_word2vec" / "models"

# общие параметры обучения; workers=1 - чтобы результат точно повторялся при том же seed
COMMON = {"min_count": 3, "epochs": 30, "seed": 42, "workers": 1, "negative": 5, "sample": 1e-3}

CONFIGS = [
    {"name": f"{'sg' if sg else 'cbow'}_w{window}_d{size}", "sg": sg, "window": window, "vector_size": size}
    for sg in (0, 1) for window in (2, 5) for size in (50, 100)
]


def train_model(sentences, config):
    params = {k: v for k, v in config.items() if k != "name"}
    return Word2Vec(sentences=sentences, **params, **COMMON)


def model_path(name):
    return MODELS_DIR / f"{name}.model"


def train_all(sentences):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for config in CONFIGS:
        model = train_model(sentences, config)
        model.save(str(model_path(config["name"])))
        models[config["name"]] = model
    return models


def load_models():
    return {c["name"]: Word2Vec.load(str(model_path(c["name"]))) for c in CONFIGS if model_path(c["name"]).exists()}


def reproducibility_check(sentences, config):
    """Обучаем одну конфигурацию дважды и сравниваем векторы."""
    first, second = train_model(sentences, config), train_model(sentences, config)
    diff = np.abs(first.wv.vectors - second.wv.vectors).max()
    return {"config": config["name"], "max_abs_diff": float(diff), "identical": bool(diff == 0)}
