




# -*- coding: utf-8 -*-
"""
🚀 Rocket Queen V7 — IA Lucky Jet
PARTIE 1 — VERSION CORRIGÉE

Modules intégrés :
  MODULE 1  — Intelligent Feedback
  MODULE 2  — Smart Memory
  MODULE 3  — Memory-Based Learning
  MODULE 4  — Feedback → Experience Memory Update
  MODULE 5  — Memory-Aware AI Score
  MODULE 6  — Experience Memory Retrieval
  MODULE 7  — Meta Decision Engine
  MODULE 8  — Adaptive Weight Learning
  MODULE 9  — Adaptive Ensemble Decision Engine
  MODULE 10 — Experience Reliability Engine

IMPORTANT :
Cette partie prépare une architecture dans laquelle le délai
de prédiction peut être appris séparément du score de cote.

Les délais supportés sont :
    1 minute
    2 minutes
    3 minutes

Le système ne doit donc pas déduire automatiquement :
"score élevé = 3 minutes".

Le délai devra être déterminé ensuite à partir :
    - du contexte,
    - des patterns,
    - de la mémoire d'expérience,
    - des résultats précédents,
    - du feedback,
    - des statistiques temporelles,
    - de la fiabilité des modèles.
"""

import os
import re
import datetime
import pickle
import uuid
import threading
import math
import json
import logging
import time
import hashlib
import random
import shutil
import html as html_mod

# Optional advanced ML backends. The bot remains runnable when these
# packages are unavailable; in that case the statistical/legacy engines
# remain active instead of pretending an unavailable model is trained.
try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.cluster import KMeans
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = HistGradientBoostingClassifier = KMeans = MLPClassifier = None
    StandardScaler = make_pipeline = None
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    # LightGBM's wheel needs OpenMP. On Nix, gcc ships libgomp but it may not
    # be in the dynamic loader's search path, so preload it when available.
    import ctypes
    import subprocess

    _libgomp = subprocess.run(
        ["gcc", "-print-file-name=libgomp.so.1"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    _libgomp_path = _libgomp.stdout.strip()
    if _libgomp.returncode == 0 and os.path.isabs(_libgomp_path):
        ctypes.CDLL(_libgomp_path, mode=ctypes.RTLD_GLOBAL)

    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

# ══════════════════════════════════════════════
# CAMEROON TIME — WAT / UTC+1
# ══════════════════════════════════════════════

os.environ["TZ"] = "Africa/Douala"

# tzset n'existe pas sur certains environnements Windows.
# On évite donc de faire planter le programme au démarrage.
try:
    time.tzset()
except AttributeError:
    pass

from dataclasses import dataclass, asdict
from typing import Optional

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract
import numpy as np
import telebot

from telebot.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from flask import Flask, request


# ══════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    os.getenv("TELEGRAM_BOT_TOKEN", "")
)

if not BOT_TOKEN:
    raise RuntimeError(
        "Le secret BOT_TOKEN est requis pour démarrer le bot Telegram."
    )

BOT_NAME = "Rocket Queen V7"
PRIMARY_SCRIPT_ID = "rocket_queen_v7_adaptive"
PRIMARY_SCRIPT_VERSION = 7.0
LEGACY_SCRIPT_IDS = {
    "lucky_rocket_ultra_ia_v6",
    "rocket_queen_v6_final",
    "rocket_queen_v6_final_adaptive",
}

bot = telebot.TeleBot(BOT_TOKEN)
flask_app = Flask(__name__)

# Rocket Queen owns a dedicated data namespace. Existing root-level data is
# copied (never moved or deleted) once, so the migration is reversible and
# another bot in the project directory cannot overwrite these active memories.
DATA_DIR = os.path.abspath(os.getenv("ROCKET_QUEEN_DATA_DIR", "data_rocket_queen_v6"))
TEMP_DIR = os.path.join(DATA_DIR, "temp")
ARCHIVES_DIR = os.path.join(DATA_DIR, "archives")
AI_STATE_FILE = os.path.join(DATA_DIR, "ai_state.json")

LEGACY_DATA_FILES = (
    "ai_state.json",
    "pattern_memory.pkl",
    "user_history.pkl",
    "feedback.pkl",
    "feedback_history.pkl",
    "pending.pkl",
    "pending_predictions.pkl",
    "feedback_samples.pkl",
    "experience_memory.pkl",
    "training_dataset.pkl",
    "history.pkl",
    "delay_memory.pkl",
    "delay_engine.pkl",
    "hour_delay_memory.pkl",
    "delay_by_mode.pkl",
    "realtime_delay_log.pkl",
    "high_cote_observations.pkl",
    "advanced_pattern_memory.pkl",
    "pattern_stats.pkl",
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ARCHIVES_DIR, exist_ok=True)

# Safe, non-destructive migration from prior versioned data folders. A folder
# with a known foreign owner is never imported. If owner metadata is absent,
# the user must explicitly opt in with ROCKET_QUEEN_IMPORT_UNVERIFIED_DATA=1.
for _legacy_dir_name in ("data_v8", "data_v7"):
    _legacy_dir = os.path.abspath(_legacy_dir_name)
    if not os.path.isdir(_legacy_dir) or _legacy_dir == DATA_DIR:
        continue
    _legacy_dir_state = os.path.join(_legacy_dir, "ai_state.json")
    _legacy_dir_id = None
    if os.path.isfile(_legacy_dir_state):
        try:
            with open(_legacy_dir_state, "r", encoding="utf-8") as _legacy_dir_state_file:
                _legacy_dir_doc = json.load(_legacy_dir_state_file)
            _legacy_dir_id = _legacy_dir_doc.get("primary_script_id")
        except Exception as _migration_error:
            logging.warning("[DATA MIGRATION] impossible de lire %s : %s", _legacy_dir_state, _migration_error)
    _legacy_dir_owner_ok = bool(
        _legacy_dir_id and (
            _legacy_dir_id in LEGACY_SCRIPT_IDS
            or _legacy_dir_id == PRIMARY_SCRIPT_ID
            or str(_legacy_dir_id).startswith(("rocket_queen", "lucky_rocket_ultra_ia"))
        )
    )
    _legacy_dir_opt_in = os.getenv("ROCKET_QUEEN_IMPORT_UNVERIFIED_DATA", "0").strip().lower() in ("1", "true", "yes", "oui")
    if not _legacy_dir_owner_ok and not _legacy_dir_opt_in:
        if any(os.path.isfile(os.path.join(_legacy_dir, name)) for name in LEGACY_DATA_FILES):
            logging.warning(
                "[DATA MIGRATION] %s existe mais son propriétaire n'est pas confirmé; aucune donnée copiée. "
                "Après vérification, définir ROCKET_QUEEN_IMPORT_UNVERIFIED_DATA=1 pour autoriser la copie non destructive.",
                _legacy_dir_name,
            )
        continue
    for _legacy_name in LEGACY_DATA_FILES:
        _legacy_source = os.path.join(_legacy_dir, _legacy_name)
        _isolated_target = os.path.join(DATA_DIR, _legacy_name)
        if os.path.isfile(_legacy_source) and not os.path.exists(_isolated_target):
            try:
                shutil.copy2(_legacy_source, _isolated_target)
                logging.info("[DATA MIGRATION] Copie non destructive : %s -> %s", _legacy_source, _isolated_target)
            except Exception as _migration_error:
                logging.warning("[DATA MIGRATION] %s : %s", _legacy_name, _migration_error)

# If the old state explicitly belongs to another script, do not import any
# root-level pickle files: they have no owner metadata and may belong to it.
_legacy_migration_allowed = True
_legacy_state_path = os.path.abspath("ai_state.json")
if os.path.isfile(_legacy_state_path):
    try:
        with open(_legacy_state_path, "r", encoding="utf-8") as _legacy_state_file:
            _legacy_state_doc = json.load(_legacy_state_file)
        _legacy_state_id = _legacy_state_doc.get("primary_script_id")
        if _legacy_state_id and _legacy_state_id not in LEGACY_SCRIPT_IDS | {PRIMARY_SCRIPT_ID}:
            _legacy_migration_allowed = False
            logging.warning(
                "[DATA MIGRATION] état racine appartenant à un autre script (%s); aucune mémoire racine copiée.",
                _legacy_state_id,
            )
    except Exception as _migration_error:
        logging.warning("[DATA MIGRATION] impossible de vérifier ai_state.json : %s", _migration_error)

if _legacy_migration_allowed:
    for _legacy_name in LEGACY_DATA_FILES:
        _legacy_source = os.path.abspath(_legacy_name)
        _isolated_target = os.path.join(DATA_DIR, _legacy_name)
        if os.path.isfile(_legacy_source) and not os.path.exists(_isolated_target):
            try:
                shutil.copy2(_legacy_source, _isolated_target)
                logging.info("[DATA MIGRATION] Copie non destructive : %s -> %s", _legacy_source, _isolated_target)
            except Exception as _migration_error:
                logging.warning("[DATA MIGRATION] %s : %s", _legacy_name, _migration_error)

# Backward compatibility for the previous pending-prediction filename.
_pending_canonical = os.path.join(DATA_DIR, "pending_predictions.pkl")
_pending_legacy = os.path.join(DATA_DIR, "pending.pkl")
if not os.path.exists(_pending_canonical) and os.path.exists(_pending_legacy):
    try:
        shutil.copy2(_pending_legacy, _pending_canonical)
    except Exception as _migration_error:
        logging.warning("[DATA MIGRATION] pending_predictions.pkl : %s", _migration_error)


# ══════════════════════════════════════════════
# ANALYSE / PREDICTION CONFIG
# ══════════════════════════════════════════════

ANALYSE_LAST = 30
MAX_PATTERNS = 100

SEQ_WEIGHTS = {
    5: 0.30,
    10: 0.35,
    15: 0.35,
}

PATTERN_THRESHOLD = 0.55

TARGET_RANGES = {
    "2X-10X": (2.0, 10.0),
    "grosse": (2.0, 30.0),
    "ultra": (5.0, 100.0),
}


# ══════════════════════════════════════════════
# IMPORTANT — DELAIS SUPPORTÉS
# ══════════════════════════════════════════════
#
# Le système apprend entre 1 et 3 minutes.
#
# Le score de prédiction ne doit PAS imposer directement
# un délai fixe.
#
# Le délai sera ensuite déterminé par les modules temporels.
# ══════════════════════════════════════════════

# ============================================================
# TIMING DYNAMIQUE v6.0
# ============================================================
#
# Le moteur temporel peut maintenant apprendre :
#   1 minute
#   2 minutes
#   3 minutes

#
# IMPORTANT :
# Le score directionnel ne choisit PAS directement la minute.
# La minute est déterminée par le moteur temporel + mémoire.
# ============================================================

SUPPORTED_DELAYS = (1, 2, 3)

# ══════════════════════════════════════════════════════════════════════
# v6 — PROBABILISTIC / TEMPORAL DECISION LAYER
# ══════════════════════════════════════════════════════════════════════
# The v6 layer deliberately sits above the existing engines.  It does not
# replace learned models with arbitrary constants; it combines their
# already-produced evidence, measures disagreement/novelty, builds a
# calibrated *decision aid*, and can abstain when the evidence is weak.
V6_ABSTAIN_ENABLED = False
V6_ABSTAIN_THRESHOLD = 0.46
V6_MIN_MODEL_AGREEMENT = 0.42
V6_MAX_NOVELTY = 0.78
V6_TIMING_MIN_CONFIDENCE = 0.43
V6_TIMING_INTERVAL_COVERAGE = 0.80
V6_RECENCY_HALF_LIFE_DAYS = 30.0
V6_LEDGER_MAX = 5000
V6_BACKTEST_MIN_SAMPLES = 20
V6_FUSION_MIN_WEIGHT = 0.05
V6_FUSION_MAX_WEIGHT = 0.35
V6_MAGNITUDE_MIN_SAMPLES = 8



# ══════════════════════════════════════════════
# FEATURE VERSION
# ══════════════════════════════════════════════

FEATURE_VERSION = 6


# ══════════════════════════════════════════════
# MARKET CONTEXTS
# ══════════════════════════════════════════════

CONTEXTS = [
    "CALM",
    "COMPRESSED",
    "VOLATILE",
    "EXPLOSIVE",
    "POST_SPIKE",
]


# ══════════════════════════════════════════════
# MODEL NAMES
# ══════════════════════════════════════════════

MODEL_NAMES = [
    "historical",
    "context",
    "pattern",
    "sequence",
    "neural",
]


META_MODEL_NAMES = [
    "markov",
    "bayes",
    "pattern",
    "neural",
    "memory",
    "historical",
]


# ══════════════════════════════════════════════
# EXPERIENCE RELIABILITY ENGINE
# ══════════════════════════════════════════════

EXP_RELIABILITY_GAIN = 0.04
EXP_RELIABILITY_LOSS = 0.05

EXP_RELIABILITY_MIN = 0.10
EXP_RELIABILITY_MAX = 1.00
EXP_RELIABILITY_DEFAULT = 0.70


# ══════════════════════════════════════════════
# MEMORY PARAMETERS
# ══════════════════════════════════════════════

MEMORY_MIN_SAMPLES = 5
MEMORY_HIGH_CONFIDENCE_SAMPLES = 15

MEMORY_MAX_INFLUENCE = 0.25
MEMORY_DISTANCE_THRESHOLD = 0.60
MEMORY_SUCCESS_NEUTRAL = 0.50


MEMORY_MIN_SIMILARITY = 0.20
MEMORY_MIN_SAMPLES_M5 = 3
MEMORY_MAX_WEIGHT = 0.30
MEMORY_CONFIDENCE_THRESHOLD = 0.50


# ══════════════════════════════════════════════
# MEMORY LIMITS
# ══════════════════════════════════════════════

MEMORY_VERSION = 3
MAX_EXPERIENCE_MEMORY = 10000
MAX_SIMILAR_EXPERIENCES = 50

MAX_FEEDBACK_SAMPLES = 10000
MAX_HIGH_COTE_OBSERVATIONS = 10000

# Prediction Ledger v6: stores only compact, pre-outcome decision metadata.
# It is intentionally bounded so a long-running bot cannot grow without limit.
prediction_ledger = []



# ══════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════

@dataclass
class FeedbackSample:

    sample_id: str

    chat_id: int
    mode: str

    predicted_target: float
    predicted_assurance: float
    predicted_reliability: float
    predicted_score: float

    prediction_datetime: str

    predicted_hour: int
    predicted_minute: int
    predicted_second: int

    actual_datetime: Optional[str]

    actual_hour: Optional[int]
    actual_minute: Optional[int]
    actual_second: Optional[int]

    delay_seconds: Optional[float]
    delay_minutes: Optional[float]

    turns_between: Optional[int]

    low_cote_losses: int

    status: str

    max_cote_reached: float

    mean_cote: float
    std_cote: float
    volatility: float

    consecutive_low: int
    peak_interval: float

    sequence: list

    created_at: str

    # Explicit linkage to the exact prediction that generated this sample.
    # Default keeps older constructor calls source-compatible.
    prediction_id: Optional[str] = None


@dataclass
class ExperienceMemory:

    experience_id: str

    timestamp: str

    hour: int
    minute: int

    mode: str

    cotes: list

    mean: float
    std: float

    low_streak: int

    volatility: float
    stability: float
    entropy: float

    bayes_probability: float

    markov_prediction: float
    markov_confidence: float

    pattern_score: float
    pattern_gain_score: float
    pattern_loss_score: float

    neural_score: float
    neural_confidence: float

    ai_score: float

    target: float

    predicted_delay: float
    predicted_time: str

    window_start: str
    window_end: str

    outcome: str = "pending"

    actual_max_cote: float = 0.0
    actual_time: str = ""

    actual_delay: float = 0.0
    actual_delay_sec: Optional[float] = None

    timing_error: float = 0.0
    target_error: float = 0.0

    created_at: str = ""
    updated_at: str = ""

    memory_version: int = MEMORY_VERSION

    low_cote_losses: int = 0


# ══════════════════════════════════════════════
# SIMPLE NEURAL NETWORK — PURE NUMPY
# ══════════════════════════════════════════════

class MiniMLP:
    """
    Petit réseau neuronal :
        12 entrées
        8 neurones cachés
        1 sortie

    La sortie représente une probabilité de succès.
    """

    def __init__(self, n_in=12, n_hidden=8):

        rng = np.random.default_rng(42)

        self.W1 = rng.normal(
            0,
            0.3,
            (n_in, n_hidden)
        )

        self.b1 = np.zeros(n_hidden)

        self.W2 = rng.normal(
            0,
            0.3,
            (n_hidden, 1)
        )

        self.b2 = np.zeros(1)

        self.n_trained = 0


    def _sigmoid(self, x):

        x = np.clip(
            x,
            -15,
            15
        )

        return 1.0 / (1.0 + np.exp(-x))


    def predict_proba(self, x: np.ndarray) -> float:

        x = np.asarray(
            x,
            dtype=float
        )

        if x.ndim != 1:
            x = x.reshape(-1)

        if len(x) != self.W1.shape[0]:
            raise ValueError(
                f"MiniMLP attend {self.W1.shape[0]} features, "
                f"reçu {len(x)}"
            )

        h = self._sigmoid(
            x @ self.W1 + self.b1
        )

        output = self._sigmoid(
            h @ self.W2 + self.b2
        )

        value = float(output[0])

        if not math.isfinite(value):
            return 0.5

        return clamp(
            value,
            0.0,
            1.0
        )


    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr=0.02
    ):

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        if len(X) == 0:
            return

        h = self._sigmoid(
            X @ self.W1 + self.b1
        )

        out = self._sigmoid(
            h @ self.W2 + self.b2
        )

        d_out = (
            out -
            y.reshape(-1, 1)
        )

        dW2 = (
            h.T @ d_out /
            len(X)
        )

        db2 = d_out.mean(
            axis=0
        )

        d_h = (
            d_out @ self.W2.T
        ) * h * (1 - h)

        dW1 = (
            X.T @ d_h /
            len(X)
        )

        db1 = d_h.mean(
            axis=0
        )

        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        self.n_trained += len(X)


_nn_model = MiniMLP()
_model_training_lock = threading.Lock()

_calibrator_a = 8.0
_calibrator_b = 0.5

_calibrator_trained = False
_neural_validation_metrics = {"brier": None, "logloss": None, "accuracy": None, "ece": None, "n_train": 0, "n_validation": 0, "validated": False}


# ══════════════════════════════════════════════
# NEURAL PREDICTION
# ══════════════════════════════════════════════

def predict_with_model(features: list) -> float:

    try:

        x = np.asarray(
            features,
            dtype=float
        )

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        return _nn_model.predict_proba(x)

    except Exception as e:

        logging.warning(
            f"[NN] Prediction failed: {e}"
        )

        return 0.5


def neural_confidence() -> float:
    metrics = _neural_validation_metrics
    n_val = int(metrics.get("n_validation", 0) or 0)
    if not metrics.get("validated") or n_val < 10:
        return 0.0
    brier = safe_float(metrics.get("brier"), 0.25)
    accuracy = clamp(safe_float(metrics.get("accuracy"), 0.50), 0.0, 1.0)
    quality = clamp(1.0 - brier / 0.25, 0.0, 1.0)
    sample_factor = clamp(n_val / 100.0, 0.0, 1.0)
    return float(clamp((0.60 * quality + 0.40 * accuracy) * sample_factor, 0.0, 1.0))


def _calibrate(score: float) -> float:

    score = clamp(
        safe_float(score, 0.5),
        0.0,
        1.0
    )

    if not _calibrator_trained:
        return score

    try:

        value = 1.0 / (
            1.0 +
            math.exp(
                -_calibrator_a *
                (score - _calibrator_b)
            )
        )

        if not math.isfinite(value):
            return score

        return clamp(
            float(value),
            0.0,
            1.0
        )

    except Exception:
        return score


# ══════════════════════════════════════════════
# NEURAL TRAINING
# ══════════════════════════════════════════════

def _nn_metrics(y_true, probs):
    y_true = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y_true) == 0:
        return {"brier": 0.25, "logloss": math.log(2.0), "accuracy": 0.5, "ece": 1.0}
    brier = float(np.mean((probs - y_true) ** 2))
    logloss = float(-np.mean(y_true*np.log(probs)+(1-y_true)*np.log(1-probs)))
    accuracy = float(np.mean((probs >= 0.5) == (y_true >= 0.5)))
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = min(1.0, lo + 0.1)
        mask = (probs >= lo) & (probs <= hi if hi >= 1.0 else probs < hi)
        if np.any(mask):
            ece += float(np.mean(mask))*abs(float(np.mean(probs[mask]))-float(np.mean(y_true[mask])))
    return {"brier": brier, "logloss": logloss, "accuracy": accuracy, "ece": ece}


def _train_model_on_dataset_impl(dataset: list):
    global _calibrator_a, _calibrator_b, _calibrator_trained, _neural_validation_metrics
    if not isinstance(dataset, list) or len(dataset) < 40:
        return
    try:
        valid=[]
        for item in dataset[-500:]:
            if not isinstance(item,dict) or item.get("feature_version",0)!=FEATURE_VERSION:
                continue
            f=item.get("features",[])
            if not isinstance(f,list) or len(f)!=12:
                continue
            y=safe_float(item.get("label"),-1)
            if y not in (0.0,1.0):
                continue
            valid.append(([clamp(safe_float(x),0,1) for x in f],y))
        if len(valid)<40: return
        X=np.asarray([x for x,_ in valid],dtype=float); y=np.asarray([z for _,z in valid],dtype=float)
        split=int(len(X)*0.80)
        if split<30 or len(X)-split<10: return
        Xtr,Xv=X[:split],X[split:]; ytr,yv=y[:split],y[split:]
        if len(np.unique(ytr))<2 or len(np.unique(yv))<2:
            logging.info("[NN] validation skipped: insufficient class diversity")
            return
        candidate=MiniMLP(); rng=np.random.default_rng(42)
        for _ in range(15):
            idx=rng.permutation(len(Xtr))
            for i in range(0,len(Xtr),32):
                b=idx[i:i+32]
                if len(b): candidate.train_step(Xtr[b],ytr[b])
        pv=np.asarray([candidate.predict_proba(x) for x in Xv])
        metrics=_nn_metrics(yv,pv)
        baseline=_nn_metrics(yv,np.full(len(yv),float(np.mean(ytr))))
        # Compare with the model already in service as well as a constant baseline.
        # A retrain is not deployed just because it trained successfully.
        previous_metrics=None
        previous_is_trained=bool(getattr(_nn_model,"n_trained",0))
        if previous_is_trained:
            try:
                previous_probs=np.asarray([_nn_model.predict_proba(x) for x in Xv])
                previous_metrics=_nn_metrics(yv,previous_probs)
            except Exception as previous_error:
                logging.warning("[NN] previous-model validation failed: %s", previous_error)
        baseline_ok=(metrics["brier"] <= baseline["brier"]+0.005 and metrics["logloss"] <= baseline["logloss"]+0.10)
        previous_ok=(not previous_is_trained or previous_metrics is None or (metrics["brier"] <= previous_metrics["brier"]+0.002 and metrics["logloss"] <= previous_metrics["logloss"]+0.05))
        deploy=bool(baseline_ok and previous_ok)
        if deploy:
            _nn_model.W1=candidate.W1.copy(); _nn_model.b1=candidate.b1.copy(); _nn_model.W2=candidate.W2.copy(); _nn_model.b2=candidate.b2.copy(); _nn_model.n_trained+=candidate.n_trained
        _neural_validation_metrics={**metrics,"baseline_brier":baseline["brier"],"baseline_logloss":baseline["logloss"],"previous_model":previous_metrics,"n_train":len(Xtr),"n_validation":len(Xv),"validated":bool(deploy)}
        pos=float(np.mean(yv)); pm=float(np.mean(pv))
        if deploy and 0<pos<1 and math.isfinite(pm):
            _calibrator_b=clamp(0.5+0.5*(pm-pos),0.15,0.85); _calibrator_trained=True
        logging.info("[NN] chronological validation train=%d val=%d deploy=%s brier=%.4f logloss=%.4f acc=%.3f ece=%.4f previous=%s",len(Xtr),len(Xv),deploy,metrics["brier"],metrics["logloss"],metrics["accuracy"],metrics["ece"],previous_metrics)
    except Exception as e:
        logging.warning("[NN] Training failed: %s",e)


def _train_model_on_dataset(dataset: list):
    """Single serialized entry point for neural retraining."""
    _model_training_lock.acquire()
    try:
        return _train_model_on_dataset_impl(dataset)
    finally:
        _model_training_lock.release()


# ══════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════

def clamp(v, lo, hi):

    try:
        value = float(v)
    except (
        TypeError,
        ValueError
    ):
        return float(lo)

    if not math.isfinite(value):
        return float(lo)

    return max(
        float(lo),
        min(
            float(hi),
            value
        )
    )


def safe_float(
    value,
    default=0.0
):

    if value is None:
        return default

    try:

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


def clean_numeric_sequence(
    arr,
    minimum=None,
    maximum=None
):

    result = []

    for value in arr or []:

        try:

            x = float(value)

            if not math.isfinite(x):
                continue

            if minimum is not None and x < minimum:
                continue

            if maximum is not None and x > maximum:
                continue

            result.append(x)

        except (
            TypeError,
            ValueError
        ):
            continue

    return result


def mean_std(arr):

    values = clean_numeric_sequence(arr)

    if not values:
        return 0.0, 0.0

    a = np.asarray(
        values,
        dtype=float
    )

    return (
        float(np.mean(a)),
        float(np.std(a))
    )


def volatility(arr) -> float:

    m, s = mean_std(arr)

    if m <= 0:
        return 0.0

    return clamp(
        s / max(m, 0.1),
        0.0,
        1.0
    )


def stability_score(arr) -> float:

    vol = volatility(arr)

    try:

        result = math.exp(
            -vol
        )

        return clamp(
            result,
            0.0,
            1.0
        )

    except Exception:

        return 0.0


# ══════════════════════════════════════════════
# ENTROPY — CORRIGÉE
# ══════════════════════════════════════════════

def entropy_score(
    arr,
    bins=8
) -> float:

    values = clean_numeric_sequence(
        arr
    )

    if len(values) < 5:
        return 0.0

    try:

        hist, _ = np.histogram(
            values,
            bins=bins
        )

        total = float(
            np.sum(hist)
        )

        if total <= 0:
            return 0.0

        # IMPORTANT :
        # on transforme les fréquences en probabilités.
        probabilities = (
            hist.astype(float) /
            total
        )

        probabilities = probabilities[
            probabilities > 0
        ]

        if len(probabilities) == 0:
            return 0.0

        ent = -float(
            np.sum(
                probabilities *
                np.log2(probabilities)
            )
        )

        max_ent = math.log2(
            bins
        )

        if max_ent <= 0:
            return 0.0

        return clamp(
            ent / max_ent,
            0.0,
            1.0
        )

    except Exception:

        return 0.0


# ══════════════════════════════════════════════
# BAYES SIMPLE
# ══════════════════════════════════════════════

def bayes_probability(
    arr,
    threshold=2.0
) -> float:

    values = clean_numeric_sequence(
        arr
    )

    if not values:
        return 0.0

    return clamp(
        sum(
            1
            for x in values
            if x >= threshold
        ) / len(values),
        0.0,
        1.0
    )


# ══════════════════════════════════════════════
# MARKOV
# ══════════════════════════════════════════════

def weighted_markov_next(
    arr,
    order=2
) -> float:

    values = clean_numeric_sequence(
        arr
    )

    if len(values) <= order:
        return 0.0

    transitions = {}

    for i in range(
        len(values) - order
    ):

        key = tuple(
            round(x, 3)
            for x in values[
                i:i + order
            ]
        )

        transitions.setdefault(
            key,
            []
        ).append(
            values[i + order]
        )


    last_key = tuple(
        round(x, 3)
        for x in values[-order:]
    )

    nexts = transitions.get(
        last_key,
        []
    )

    if not nexts:

        return float(
            np.mean(values)
        )

    return float(
        np.mean(nexts)
    )


# ══════════════════════════════════════════════
# CONSECUTIVE LOW
# ══════════════════════════════════════════════

def consecutive_low(
    arr,
    threshold=2.0
) -> int:

    values = clean_numeric_sequence(
        arr
    )

    count = 0

    for x in reversed(values):

        if x < threshold:
            count += 1
        else:
            break

    return count


# ══════════════════════════════════════════════
# PEAK INTERVAL
# ══════════════════════════════════════════════

def compute_peak_interval(
    arr,
    threshold=10.0
) -> Optional[float]:

    values = clean_numeric_sequence(
        arr
    )

    indices = [
        i
        for i, x in enumerate(values)
        if x >= threshold
    ]

    if len(indices) < 2:
        return None

    return float(
        np.mean(
            np.diff(indices)
        )
    )


# ══════════════════════════════════════════════
# LEARNED SCORE
# ══════════════════════════════════════════════

def compute_learned_score(
    low,
    stability,
    entropy,
    bayes,
    markov,
    pattern
) -> float:

    score = (

        0.30 * clamp(low, 0.0, 1.0)

        + 0.15 *
        clamp(stability, 0.0, 1.0)

        + 0.10 *
        clamp(entropy, 0.0, 1.0)

        + 0.20 *
        clamp(bayes, 0.0, 1.0)

        + 0.15 *
        clamp(markov, 0.0, 1.0)

        + 0.10 *
        clamp(pattern, 0.0, 1.0)
    )

    return clamp(
        score,
        0.0,
        1.0
    )


# ══════════════════════════════════════════════════════════════════════
# v6 — DECISION INTELLIGENCE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _v6_safe_probability(value, default=0.50):
    """Convert a model score/probability to [0,1] without inventing data."""
    try:
        x = float(value)
    except Exception:
        x = float(default)
    if not math.isfinite(x):
        x = float(default)
    if x > 1.0:
        x /= 100.0
    return clamp(x, 0.0, 1.0)


def _v6_entropy(probabilities):
    vals = [_v6_safe_probability(v, 0.0) for v in probabilities]
    vals = [v for v in vals if v > 0.0]
    if not vals:
        return 1.0
    z = sum(vals)
    if z <= 0:
        return 1.0
    vals = [v / z for v in vals]
    h = -sum(v * math.log(v) for v in vals)
    return clamp(h / math.log(max(len(vals), 2)), 0.0, 1.0)


def v6_model_agreement(values):
    """Agreement is high when model probabilities cluster tightly."""
    vals = [_v6_safe_probability(v, 0.50) for v in values]
    if len(vals) < 2:
        return 0.50
    mean_v = sum(vals) / len(vals)
    variance = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    # Bernoulli probability scores live in [0,1], so variance <= .25.
    return clamp(1.0 - math.sqrt(variance) / 0.5, 0.0, 1.0)


def v6_recency_weight(timestamp, half_life_days=V6_RECENCY_HALF_LIFE_DAYS):
    """Exponential recency weight. Invalid timestamps receive a neutral weight."""
    try:
        dt = datetime.datetime.fromisoformat(str(timestamp))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        age = max(0.0, (datetime.datetime.now() - dt).total_seconds() / 86400.0)
        return 0.5 ** (age / max(float(half_life_days), 1e-6))
    except Exception:
        return 0.50


def v6_experience_novelty(last, mode):
    """Estimate novelty from verified historical experiences only.

    This is intentionally conservative: absence of memory is not treated as
    evidence of success.  It only lowers confidence.
    """
    if not last or not experience_memory:
        return 1.0, 0, 0.0
    try:
        current = build_current_context(last, mode, 0.0)
    except Exception:
        current = {}
    similarities = []
    for exp in experience_memory:
        try:
            if str(exp.get("outcome", "pending")).lower() == "pending":
                continue
            sim = _timing_experience_similarity(current, exp, mode)
            if sim > 0:
                w = sim * v6_recency_weight(exp.get("timestamp", ""))
                similarities.append((w, sim))
        except Exception:
            continue
    if not similarities:
        return 1.0, 0, 0.0
    similarities.sort(reverse=True)
    top = similarities[:MAX_SIMILAR_EXPERIENCES]
    support = sum(w for w, _ in top)
    weighted_sim = (sum(w * sim for w, sim in top) / support) if support > 0 else 0.0
    # More close historical support => lower novelty.
    support_factor = 1.0 - math.exp(-support / 5.0)
    novelty = 1.0 - (weighted_sim * support_factor)
    return clamp(novelty, 0.0, 1.0), len(top), weighted_sim


def v6_timing_interval(delay_probabilities, coverage=V6_TIMING_INTERVAL_COVERAGE):
    """Smallest contiguous delay-window interval containing target coverage."""
    if not delay_probabilities:
        return None
    items = sorted((int(k), _v6_safe_probability(v, 0.0)) for k, v in delay_probabilities.items())
    total = sum(v for _, v in items)
    if total <= 0:
        return None
    items = [(k, v / total) for k, v in items]
    best = None
    for i in range(len(items)):
        acc = 0.0
        for j in range(i, len(items)):
            acc += items[j][1]
            if acc >= coverage:
                width = items[j][0] - items[i][0]
                candidate = (width, items[i][0], items[j][0], acc)
                if best is None or candidate < best:
                    best = candidate
                break
    if best is None:
        return None
    _, lo, hi, covered = best
    return {"low_min": lo, "high_min": hi + 1, "coverage": round(covered, 4)}


def v6_timing_entropy(delay_probabilities):
    vals = list((delay_probabilities or {}).values())
    return _v6_entropy(vals)


def v6_ensemble_direction(last, ai_score, pattern_result, bayes, markov, stability, entropy):
    """Four-way directional ensemble from already-computed evidence.

    This is a probability-like score, not a claim that an unknown event is
    guaranteed.  It deliberately discounts high entropy and rewards agreement.
    """
    pat = pattern_result if isinstance(pattern_result, dict) else {}
    pattern = _v6_safe_probability(pat.get("confidence", pat.get("score", pat.get("pattern_score", 0.50))))
    components = {
        "ai": _v6_safe_probability(ai_score),
        "bayes": _v6_safe_probability(bayes),
        "markov": _v6_safe_probability(markov),
        "pattern": pattern,
    }
    mean_p = sum(components.values()) / len(components)
    agreement = v6_model_agreement(list(components.values()))
    entropy_penalty = 1.0 - _v6_safe_probability(entropy)
    stability = _v6_safe_probability(stability)
    probability = (
        0.30 * components["ai"]
        + 0.25 * components["bayes"]
        + 0.20 * components["markov"]
        + 0.15 * components["pattern"]
        + 0.10 * entropy_penalty
    )
    confidence = (
        0.45 * agreement
        + 0.25 * stability
        + 0.20 * entropy_penalty
        + 0.10 * min(1.0, len(last) / float(max(ANALYSE_LAST, 1)))
    )
    return clamp(probability, 0.0, 1.0), clamp(confidence, 0.0, 1.0), components, agreement


def v6_decision_gate(direction_probability, direction_confidence, timing_confidence,
                     model_agreement, novelty, timing_model_status):
    """Return a transparent PREDICT decision and informational reasons."""
    reasons = []
    confidence = (
        0.35 * _v6_safe_probability(direction_probability)
        + 0.25 * _v6_safe_probability(direction_confidence)
        + 0.20 * _v6_safe_probability(timing_confidence)
        + 0.15 * _v6_safe_probability(model_agreement)
        + 0.05 * (1.0 - _v6_safe_probability(novelty))
    )
    if model_agreement < V6_MIN_MODEL_AGREEMENT:
        reasons.append("model_disagreement")
    if novelty > V6_MAX_NOVELTY:
        reasons.append("high_novelty")
    if timing_confidence < V6_TIMING_MIN_CONFIDENCE:
        reasons.append("timing_uncertain")
    if timing_model_status == "untrained":
        reasons.append("timing_model_untrained")
    if direction_confidence < 0.38:
        reasons.append("direction_uncertain")
    # V6 always returns a prediction. Confidence is informational only and
    # never blocks the user from receiving target/timing output.
    decision = "PREDICT"
    return decision, clamp(confidence, 0.0, 1.0), reasons


def v6_build_prediction_ledger_entry(result, mode, cotes):
    """Create a compact pre-outcome audit record."""
    return {
        "prediction_id": result.get("prediction_id"),
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": mode,
        "direction_probability": result.get("direction_probability"),
        "direction_confidence": result.get("direction_confidence"),
        "target": result.get("target"),
        "timing_distribution": result.get("delay_probabilities", {}),
        "timing_confidence": result.get("timing_confidence"),
        "timing_interval": result.get("timing_interval"),
        "timing_model_status": result.get("timing_model_status"),
        "model_agreement": result.get("model_agreement"),
        "novelty": result.get("novelty_score"),
        "decision_confidence": result.get("decision_confidence"),
        "decision": result.get("decision"),
        "decision_reasons": list(result.get("decision_reasons", [])),
        "sample_size": len(cotes or []),
    }


def v6_record_prediction(result, mode, cotes):
    try:
        prediction_ledger.append(v6_build_prediction_ledger_entry(result, mode, cotes))
        if len(prediction_ledger) > V6_LEDGER_MAX:
            del prediction_ledger[:-V6_LEDGER_MAX]
    except Exception as exc:
        logging.warning("[V6 LEDGER] %s", exc)


def v6_backtest_direction(samples):
    """Chronological holdout report using stored training samples.

    It never trains on the validation tail.  The report is deliberately
    lightweight and dependency-free so it can run inside the bot.
    """
    valid = [x for x in (samples or []) if isinstance(x, dict) and "features" in x and "label" in x]
    valid = [x for x in valid if isinstance(x.get("features"), (list, tuple))]
    if len(valid) < V6_BACKTEST_MIN_SAMPLES:
        return {"status": "insufficient_data", "n": len(valid)}
    cut = max(1, int(len(valid) * 0.80))
    train = valid[:cut]
    test = valid[cut:]
    # Do not mutate the production network.  Use a fresh candidate.
    candidate = MiniMLP()
    X = np.asarray([x["features"] for x in train], dtype=float)
    y = np.asarray([x["label"] for x in train], dtype=float)
    for _ in range(15):
        candidate.train_step(X, y, lr=0.02)
    preds = [candidate.predict_proba(x["features"]) for x in test]
    ys = [float(x["label"]) for x in test]
    if not preds:
        return {"status": "empty_validation", "n": len(valid)}
    brier = sum((p-yv)**2 for p, yv in zip(preds, ys)) / len(preds)
    eps = 1e-7
    logloss = -sum(yv*math.log(max(p,eps)) + (1-yv)*math.log(max(1-p,eps)) for p,yv in zip(preds,ys)) / len(preds)
    accuracy = sum((p >= 0.5) == bool(yv) for p,yv in zip(preds,ys)) / len(preds)
    return {"status":"ok", "n":len(valid), "train_n":len(train), "validation_n":len(test),
            "brier":round(float(brier),6), "logloss":round(float(logloss),6), "accuracy":round(float(accuracy),6)}


def v6_backtest_directional_feedback(feedbacks=None, minimum_samples=30):
    """Evaluate the saved direction probabilities against their real feedback labels.

    This is an evaluation of historical predictions, not a newly trained model.
    The chronological tail is held out; small samples are marked descriptive only.
    """
    source = feedbacks if feedbacks is not None else feedback_history
    rows = []
    for index, item in enumerate(source or []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("learning_status", item.get("status", item.get("outcome", ""))) or "").lower().strip()
        if status not in ("gain", "perte", "loss"):
            continue
        label = 1.0 if status == "gain" else 0.0
        prediction = item.get("prediction") if isinstance(item.get("prediction"), dict) else {}
        raw_probability = item.get("direction_probability", prediction.get("direction_probability"))
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            continue
        stamp = item.get("pred_time") or item.get("predicted_datetime") or item.get("time") or index
        try:
            stamp_key = datetime.datetime.fromisoformat(str(stamp)).timestamp()
        except Exception:
            stamp_key = float(index)
        rows.append({"p": probability, "y": label, "stamp": stamp_key})
    rows.sort(key=lambda row: row["stamp"])

    def metrics(items):
        if not items:
            return {"n": 0, "accuracy": None, "brier": None, "logloss": None, "mean_probability": None, "observed_gain_rate": None}
        eps = 1e-7
        n = len(items)
        return {
            "n": n,
            "accuracy": round(sum((r["p"] >= .5) == bool(r["y"]) for r in items) / n, 4),
            "brier": round(sum((r["p"] - r["y"]) ** 2 for r in items) / n, 6),
            "logloss": round(-sum(r["y"] * math.log(max(r["p"], eps)) + (1-r["y"]) * math.log(max(1-r["p"], eps)) for r in items) / n, 6),
            "mean_probability": round(sum(r["p"] for r in items) / n, 4),
            "observed_gain_rate": round(sum(r["y"] for r in items) / n, 4),
        }

    if len(rows) < 5:
        return {"status": "insufficient_data", "n": len(rows), "minimum_for_descriptive_report": 5, "minimum_for_calibration": minimum_samples}
    cut = max(1, int(len(rows) * .80))
    train, test = rows[:cut], rows[cut:]
    if not test:
        return {"status": "insufficient_data", "n": len(rows), "minimum_for_calibration": minimum_samples}
    train_gain_rate = sum(r["y"] for r in train) / len(train)
    majority_label = 1.0 if train_gain_rate >= .5 else 0.0
    baseline_accuracy = sum(r["y"] == majority_label for r in test) / len(test)
    return {
        "status": "ok" if len(rows) >= minimum_samples else "descriptive_only_insufficient_data",
        "method": "saved_direction_probability_chronological_holdout",
        "n": len(rows), "minimum_for_calibration": minimum_samples,
        "train_n": len(train), "validation_n": len(test),
        "overall": metrics(rows), "chronological_holdout": metrics(test),
        "majority_baseline_accuracy_on_holdout": round(baseline_accuracy, 4),
        "note": "Le petit échantillon ne permet pas de conclure à une performance future."
    }


def v6_backtest_prediction_levels(samples=None, minimum_samples=30):
    """Chronological evaluation of historical target/assurance/reliability thresholds.

    It evaluates the values actually displayed against the recorded maximum cote;
    it does not pretend that multiplier values are probabilities. With fewer than
    minimum_samples it reports descriptive results but marks the sample insufficient.
    """
    source = samples if samples is not None else feedback_samples
    rows = []
    for index, item in enumerate(source or []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", item.get("learning_status", ""))).lower().strip()
        if status not in ("gain", "perte", "loss"):
            continue
        actual = _safe_float(item.get("max_cote_reached", item.get("actual_max_cote", item.get("max_cote"))), 0.0)
        target = _safe_float(item.get("predicted_target", item.get("target")), 0.0)
        assurance = _safe_float(item.get("predicted_assurance", item.get("assurance")), 0.0)
        reliability = _safe_float(item.get("predicted_reliability", item.get("fiabilite", item.get("reliability"))), 0.0)
        if actual <= 0 or target <= 0 or assurance <= 0 or reliability <= 0:
            continue
        stamp = item.get("prediction_datetime") or item.get("pred_time") or item.get("created_at") or index
        try:
            stamp_key = datetime.datetime.fromisoformat(str(stamp)).timestamp()
        except Exception:
            stamp_key = float(index)
        rows.append({"actual": actual, "target": target, "assurance": assurance, "reliability": reliability, "stamp": stamp_key})
    rows.sort(key=lambda row: row["stamp"])

    def metrics(items):
        if not items:
            return {"n": 0, "target_hit_rate": None, "assurance_hit_rate": None, "reliability_hit_rate": None, "mean_target_shortfall": None}
        n = len(items)
        return {
            "n": n,
            "target_hit_rate": round(sum(r["actual"] >= r["target"] for r in items) / n, 4),
            "assurance_hit_rate": round(sum(r["actual"] >= r["assurance"] for r in items) / n, 4),
            "reliability_hit_rate": round(sum(r["actual"] >= r["reliability"] for r in items) / n, 4),
            "mean_target_shortfall": round(sum(max(0.0, r["target"] - r["actual"]) for r in items) / n, 3),
        }

    if not rows:
        return {"status": "insufficient_data", "n": 0, "minimum": minimum_samples, "overall": metrics([]), "chronological_holdout": metrics([])}
    cut = max(1, int(len(rows) * .80)) if len(rows) > 1 else len(rows)
    test = rows[cut:] if cut < len(rows) else []
    return {
        "status": "ok" if len(rows) >= minimum_samples else "descriptive_only_insufficient_data",
        "method": "historical_threshold_hit_rates_chronological_holdout",
        "n": len(rows), "minimum_for_calibration": minimum_samples,
        "train_n": len(rows) - len(test), "validation_n": len(test),
        "overall": metrics(rows), "chronological_holdout": metrics(test),
        "note": "Taux descriptifs de seuil atteint; un petit échantillon ne valide pas la performance future."
    }


def v6_historical_data_quality_report(samples=None, feedbacks=None):
    """Detect data-quality issues without deleting or rewriting raw sequences."""
    sample_rows = samples if samples is not None else feedback_samples
    feedback_rows = feedbacks if feedbacks is not None else feedback_history
    repeated_halves = 0
    duplicate_ids = 0
    seen_ids = set()
    mode_counts = {}
    samples_missing_prediction_id = 0
    for item in sample_rows or []:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if not item.get("prediction_id"):
            samples_missing_prediction_id += 1
        pid = item.get("prediction_id") or item.get("sample_id")
        if pid:
            if str(pid) in seen_ids:
                duplicate_ids += 1
            seen_ids.add(str(pid))
        seq = item.get("sequence", item.get("cotes", []))
        try:
            seq = [round(float(v), 6) for v in seq]
        except Exception:
            seq = []
        if len(seq) >= 8 and len(seq) % 2 == 0 and seq[:len(seq)//2] == seq[len(seq)//2:]:
            repeated_halves += 1
    repeated_half_examples = []
    # Second pass gives examples without altering the original sequences.
    for index, item in enumerate(sample_rows or []):
        if not isinstance(item, dict):
            continue
        seq = item.get("sequence", item.get("cotes", []))
        try:
            seq = [round(float(v), 6) for v in seq]
        except Exception:
            seq = []
        if len(seq) >= 8 and len(seq) % 2 == 0 and seq[:len(seq)//2] == seq[len(seq)//2:]:
            repeated_half_examples.append({"index": index, "sample_id": item.get("sample_id"), "length": len(seq)})

    timing_anomalies = []
    timing_checked = 0
    for index, item in enumerate(feedback_rows or []):
        if not isinstance(item, dict):
            continue
        source = item.get("actual_delay_source")
        raw_delay = item.get("actual_delay_sec")
        try:
            delay = float(raw_delay) if raw_delay is not None else None
        except (TypeError, ValueError):
            delay = None
        pred_stamp = item.get("pred_time") or item.get("predicted_datetime")
        actual_stamp = item.get("actual_datetime")
        reasons = []
        if source == "observed_datetime" and delay is not None:
            timing_checked += 1
            if not math.isfinite(delay) or delay <= 0:
                reasons.append("non_positive_or_non_finite_elapsed_time")
            elif delay > V6_TIMING_MAX_OBSERVED_SEC:
                reasons.append("elapsed_time_over_30_minutes")
        # Cross-check stored elapsed seconds against timestamps when both exist.
        if pred_stamp and actual_stamp:
            try:
                pred_dt = datetime.datetime.fromisoformat(str(pred_stamp))
                actual_dt = datetime.datetime.fromisoformat(str(actual_stamp))
                delta = (actual_dt - pred_dt).total_seconds()
                if delta < 0 or delta > V6_TIMING_MAX_OBSERVED_SEC:
                    reasons.append("timestamp_delta_outside_supported_observation_window")
                if delay is not None and math.isfinite(delta) and abs(delta - delay) > 2.0:
                    reasons.append("stored_delay_disagrees_with_timestamps")
            except Exception:
                reasons.append("timestamp_parse_error")
        if reasons:
            timing_anomalies.append({
                "index": index,
                "prediction_id": item.get("prediction_id"),
                "pred_time": pred_stamp,
                "actual_datetime": actual_stamp,
                "actual_delay_sec": delay,
                "actual_delay_source": source,
                "reasons": list(dict.fromkeys(reasons)),
            })

    return {
        "samples": len(sample_rows or []),
        "feedbacks": len(feedback_rows or []),
        "mode_counts": mode_counts,
        "repeated_half_sequences_suspected": repeated_halves,
        "repeated_half_examples": repeated_half_examples[:10],
        "duplicate_prediction_or_sample_ids": duplicate_ids,
        "samples_missing_prediction_id": samples_missing_prediction_id,
        "timing_observations_checked": timing_checked,
        "invalid_legacy_timing_records": len(timing_anomalies),
        "timing_anomaly_examples": timing_anomalies[:10],
        "note": "Les séquences répétées et les délais anormaux sont signalés pour vérification; aucune donnée brute n'est supprimée automatiquement. Les délais hors fenêtre sont exclus des calibrations temporelles.",
    }


# ══════════════════════════════════════════════
# MODULE 1 — PREDICTION FEATURES
# ══════════════════════════════════════════════
#
# 12 FEATURES
#
# Aucun résultat futur ne doit entrer ici.
# ══════════════════════════════════════════════

def build_prediction_features(
    *,
    consecutive_low_value: float,
    volatility_value: float,
    std_value: float,
    mean_value: float,
    predicted_score: float,
    predicted_target: float,
    predicted_hour: int,
    predicted_minute: int,
    peak_interval: float,
    sequence_length: int,
    pattern_gain: float = 0.0,
    pattern_loss: float = 0.0,
) -> list:

    features = [

        clamp(
            safe_float(
                consecutive_low_value
            ) / 10.0,
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                volatility_value
            ),
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                std_value
            ) / 10.0,
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                mean_value
            ) / 50.0,
            0.0,
            1.0
        ),

        0.0,  # reserved legacy feature: predicted score (post-model)
        0.0,  # reserved legacy feature: predicted target (post-model)


        clamp(
            safe_float(
                predicted_hour
            ) / 23.0,
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                predicted_minute
            ) / 59.0,
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                peak_interval
            ) / 30.0,
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                sequence_length
            ) / max(
                ANALYSE_LAST,
                1
            ),
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                pattern_gain
            ),
            0.0,
            1.0
        ),

        clamp(
            safe_float(
                pattern_loss
            ),
            0.0,
            1.0
        ),
    ]

    return [
        0.0 if not math.isfinite(x)
        else float(x)
        for x in features
    ]


# ══════════════════════════════════════════════
# PATTERN DETECTION
# ══════════════════════════════════════════════

def is_similar_pattern(
    new_cotes,
    saved_patterns,
    threshold=PATTERN_THRESHOLD
):

    new_values = clean_numeric_sequence(
        new_cotes
    )

    if not new_values:
        return False, 0.0

    if not saved_patterns:
        return False, 0.0

    best = float("inf")

    for pattern in saved_patterns:

        pattern_values = clean_numeric_sequence(
            pattern
        )

        if len(pattern_values) != len(new_values):
            continue

        try:

            new_array = np.asarray(new_values, dtype=float)
            pattern_array = np.asarray(pattern_values, dtype=float)

            # Une même forme peut apparaître à une échelle différente.
            # La distance brute conserve l'information de niveau, tandis
            # que la distance normalisée reconnaît la forme de la séquence.
            raw_scale = max(
                float(np.mean(np.abs(new_array))),
                float(np.mean(np.abs(pattern_array))),
                1.0
            )
            raw_diff = (
                np.linalg.norm(new_array - pattern_array)
                / max(len(new_values), 1)
                / raw_scale
            )
            new_mean = max(float(np.mean(new_array)), 1e-9)
            pattern_mean = max(float(np.mean(pattern_array)), 1e-9)
            shape_diff = (
                np.linalg.norm(
                    (new_array / new_mean)
                    - (pattern_array / pattern_mean)
                )
                / max(len(new_values), 1)
            )
            diff = min(raw_diff, shape_diff)

            if math.isfinite(
                diff
            ):
                best = min(
                    best,
                    float(diff)
                )

        except Exception:
            continue


    if not math.isfinite(best):
        return False, 0.0


    if best < threshold:

        similarity = (
            1.0 -
            (
                best /
                max(
                    threshold,
                    1e-9
                )
            )
        )

        similarity = clamp(
            similarity,
            0.0,
            1.0
        )

        return (
            True,
            round(
                similarity * 100,
                1
            )
        )


    return False, 0.0


# ══════════════════════════════════════════════
# WEIGHTED PATTERN DETECTION
# ══════════════════════════════════════════════

def detect_weighted_patterns(
    last,
    p_memory,
    seq_weights
) -> dict:

    raw_gain = 0.0
    raw_loss = 0.0

    details = {}

    # On calcule la capacité maximale réelle
    # des contributions.
    max_possible = sum(
        max(
            0.0,
            safe_float(
                seq_weights.get(L, 0.0)
            )
        )
        for L in seq_weights
    )

    for L in sorted(
        seq_weights.keys()
    ):

        details[L] = {
            "available": False,
            "gain_match": False,
            "loss_match": False,
            "gain_sim": 0.0,
            "loss_sim": 0.0,
        }

        if len(last) < L:
            continue

        seq = last[-L:]

        gain_memory = (
            p_memory
            .get("gains", {})
            .get(L, [])
        )

        loss_memory = (
            p_memory
            .get("pertes", {})
            .get(L, [])
        )

        details[L]["available"] = True


        gain_match, gain_sim = (
            is_similar_pattern(
                seq,
                gain_memory
            )
        )

        loss_match, loss_sim = (
            is_similar_pattern(
                seq,
                loss_memory
            )
        )


        if gain_match:

            details[L]["gain_match"] = True

            details[L]["gain_sim"] = (
                gain_sim
            )

            raw_gain += (
                safe_float(
                    seq_weights.get(L, 0.0)
                )
                *
                gain_sim
                / 100.0
            )


        if loss_match:

            details[L]["loss_match"] = True

            details[L]["loss_sim"] = (
                loss_sim
            )

            raw_loss += (
                safe_float(
                    seq_weights.get(L, 0.0)
                )
                *
                loss_sim
                / 100.0
            )


    net = raw_gain - raw_loss

    if max_possible > 0:

        net_norm = (
            net /
            max_possible
        )

    else:
        net_norm = 0.0


    pattern_bonus = clamp(
        net_norm * 0.3,
        -0.3,
        0.3
    )


    return {

        "details": details,

        "raw_gain": raw_gain,

        "raw_loss": raw_loss,

        "net": pattern_bonus,

    }


# ══════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════

def _data_path(filename):
    """Resolve persisted files inside the dedicated Rocket Queen data folder."""
    filename = os.fspath(filename)
    if os.path.isabs(filename):
        return filename
    if os.path.abspath(filename).startswith(DATA_DIR + os.sep):
        return filename
    return os.path.join(DATA_DIR, filename)


def save_file(
    filename,
    obj
):

    try:
        filename = _data_path(filename)

        directory = os.path.dirname(
            filename
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        temp_filename = (
            f"{filename}.tmp"
        )

        with open(
            temp_filename,
            "wb"
        ) as f:

            pickle.dump(
                obj,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        # Remplacement atomique
        os.replace(
            temp_filename,
            filename
        )

    except Exception as e:

        logging.warning(
            f"save_file({filename}): {e}"
        )


def load_file(
    filename,
    default
):

    try:
        filename = _data_path(filename)

        if not os.path.exists(
            filename
        ):
            return default

        with open(
            filename,
            "rb"
        ) as f:

            result = pickle.load(f)

        return result

    except Exception as e:

        logging.warning(
            f"load_file({filename}): {e}"
        )

        return default


# ══════════════════════════════════════════════
# FEEDBACK MIGRATION
# ══════════════════════════════════════════════

def normalize_feedback_history(
    raw
) -> list:

    """
    Convertit l'ancien format :

        {
            "rocket": [...],
            "lucky": [...]
        }

    vers une liste uniforme.

    Les données historiques sont conservées.
    """

    if isinstance(
        raw,
        list
    ):
        return raw

    if not isinstance(
        raw,
        dict
    ):
        return []


    normalized = []

    for legacy_mode, entries in raw.items():

        if not isinstance(
            entries,
            list
        ):
            continue


        for entry in entries:

            if not isinstance(
                entry,
                dict
            ):
                continue


            item = dict(
                entry
            )


            sequence = list(
                item.get("seq")
                or
                item.get("sequence")
                or
                []
            )


            # Nettoyage de la séquence
            sequence = [
                safe_float(x)
                for x in sequence
                if math.isfinite(
                    safe_float(x)
                )
            ]


            try:

                status = (
                    "gain"
                    if safe_float(
                        item.get(
                            "label",
                            0
                        )
                    ) >= 0.5
                    else "perte"
                )

            except Exception:

                status = item.get(
                    "status",
                    "perte"
                )


            item.setdefault(
                "status",
                status
            )

            item.setdefault(
                "mode",
                legacy_mode
            )

            item.setdefault(
                "max_cote",
                safe_float(
                    item.get(
                        "real",
                        0.0
                    )
                )
            )

            item.setdefault(
                "real_time",
                None
            )

            item.setdefault(
                "delay_sec",
                None
            )

            item.setdefault(
                "cotes",
                sequence
            )

            item.setdefault(
                "prediction",
                {
                    "pattern_sequence": sequence,
                    "delay_min": None,
                    "delay_sec": None,
                }
            )

            item.setdefault(
                "time",
                item.get(
                    "ts",
                    datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                )
            )

            item["legacy_format"] = True

            normalized.append(
                item
            )


    logging.info(
        "[MIGRATION] Legacy feedback "
        f"converted: {len(normalized)} records"
    )

    return normalized


# ══════════════════════════════════════════════
# DELAY STATS NORMALIZATION
# ══════════════════════════════════════════════

def normalize_delay_stats():

    global DELAY_STATS

    for delay in SUPPORTED_DELAYS:

        bucket = DELAY_STATS.setdefault(
            delay,
            {}
        )

        defaults = {
            "total": 0,
            "success": 0,
            "loss": 0,
            "avg_score": 0.0,
        }

        for key, default in defaults.items():

            value = bucket.get(
                key,
                default
            )

            if not isinstance(
                value,
                (int, float)
            ):

                bucket[key] = default

            elif not math.isfinite(
                float(value)
            ):

                bucket[key] = default


        if not isinstance(
            bucket.get(
                "recent_results"
            ),
            list
        ):

            bucket[
                "recent_results"
            ] = []


        bucket[
            "recent_results"
        ] = bucket[
            "recent_results"
        ][-50:]


# ══════════════════════════════════════════════
# TIMING MEMORY NORMALIZATION
# ══════════════════════════════════════════════

def normalize_timing_memory():

    global AI_STATE

    memory = AI_STATE.get(
        "timing_memory"
    )


    if not isinstance(
        memory,
        dict
    ):

        AI_STATE[
            "timing_memory"
        ] = {}

        return


    for context_key, context_buckets in list(
        memory.items()
    ):

        if not isinstance(
            context_buckets,
            dict
        ):

            memory[
                context_key
            ] = {}

            continue


        for delay in SUPPORTED_DELAYS:

            key = str(delay)

            bucket = context_buckets.get(
                key
            )


            if not isinstance(
                bucket,
                dict
            ):

                # Cette fonction sera fournie
                # par la partie suivante.
                #
                # On initialise temporairement
                # un bucket minimal si elle
                # n'existe pas encore.

                context_buckets[key] = {
                    "total": 0,
                    "success": 0,
                    "loss": 0,
                    "early": 0,
                    "late": 0,
                    "recent_results": [],
                    "recent_observed_delays": [],
                }

                continue


            for field in (
                "total",
                "success",
                "loss",
                "early",
                "late",
            ):

                value = bucket.get(
                    field,
                    0
                )

                if not isinstance(
                    value,
                    (int, float)
                ):

                    bucket[field] = 0


            for field in (
                "recent_results",
                "recent_observed_delays",
            ):

                if not isinstance(
                    bucket.get(field),
                    list
                ):

                    bucket[field] = []


                bucket[field] = (
                    bucket[field][-50:]
                )


# ══════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════

pattern_memory = {
    "gains": {
        5: [],
        10: [],
        15: [],
    },

    "pertes": {
        5: [],
        10: [],
        15: [],
    },
}


user_history = {}

feedback_history = []

pending_predictions = {}

last_cotes = {}

feedback_samples = []

experience_memory = []

training_dataset = []

history = []

delay_memory = {}

delay_engine = {}

hour_delay_memory = {}

delay_by_mode = {}

realtime_delay_log = []

high_cote_observations = []

advanced_pattern_memory = {}

pattern_stats = {}

message_hashes = {}

MESSAGE_DEDUP_WINDOW = 5

# Empêche deux déclenchements du réentraînement quotidien autour de 10h.
daily_retrain_lock = threading.Lock()


def _normalize_pending_predictions(raw):
    """
    Migre l'ancien format {chat_id: prediction} vers
    {chat_id: {prediction_id: prediction}}.
    """
    if not isinstance(raw, dict):
        return {}

    normalized = {}

    for raw_chat_id, value in raw.items():
        try:
            chat_id = int(raw_chat_id)
        except (TypeError, ValueError):
            chat_id = raw_chat_id

        if not isinstance(value, dict):
            continue

        # Ancien format : la prédiction possède directement "result".
        if isinstance(value.get("result"), dict):
            prediction_id = str(
                value.get("prediction_id")
                or value.get("experience_id")
                or f"legacy-{chat_id}"
            )
            prediction = dict(value)
            prediction["prediction_id"] = prediction_id
            normalized.setdefault(chat_id, {})[prediction_id] = prediction
            continue

        # Nouveau format : plusieurs prédictions peuvent coexister.
        bucket = {}
        for raw_id, prediction in value.items():
            if not isinstance(prediction, dict):
                continue
            prediction_id = str(
                prediction.get("prediction_id")
                or raw_id
            )
            prediction_copy = dict(prediction)
            prediction_copy["prediction_id"] = prediction_id
            bucket[prediction_id] = prediction_copy

        if bucket:
            normalized[chat_id] = bucket

    return normalized


def _pending_prediction(chat_id, prediction_id=None):
    """Retourne uniquement la prédiction demandée pour ce chat."""
    bucket = pending_predictions.get(chat_id, {})
    if not isinstance(bucket, dict):
        return None

    if isinstance(bucket.get("result"), dict):
        return bucket

    if prediction_id:
        return bucket.get(str(prediction_id))

    if not bucket:
        return None

    return max(
        bucket.values(),
        key=lambda item: safe_float(item.get("created_at"), 0.0)
        if isinstance(item, dict)
        else 0.0
    )

AI_STATE = {}

meta_model_stats = {}

_last_prediction_meta = {
    "model_outputs": {},
    "meta_result": {},
    "context": "CALM",
}


# ══════════════════════════════════════════════
# DELAY COMPETITION STATS
# ══════════════════════════════════════════════

DELAY_STATS = {

    delay: {
        "total": 0,
        "success": 0,
        "loss": 0,
        "recent_results": [],
        "avg_score": 0.0,
    }

    for delay in SUPPORTED_DELAYS
}


def sanitize_legacy_timing_records():
    """Neutralise corrupted legacy elapsed times without deleting outcomes.

    Records and raw timestamps are retained for audit. Only derived timing
    measurements that are negative or exceed 30 minutes are nulled so they
    cannot contaminate timing memories or backtests. Directional labels and
    maximum-cote observations are preserved.
    """
    changed = {"feedback": 0, "experience": 0, "samples": 0}

    def parse_stamp(value):
        try:
            return datetime.datetime.fromisoformat(str(value)) if value else None
        except Exception:
            return None

    for record in feedback_history if isinstance(feedback_history, list) else []:
        if not isinstance(record, dict):
            continue
        pred_dt = parse_stamp(record.get("pred_time") or record.get("predicted_datetime"))
        actual_dt = parse_stamp(record.get("actual_datetime"))
        raw_delay = record.get("actual_delay_sec")
        try:
            raw_delay = float(raw_delay) if raw_delay is not None else None
        except (TypeError, ValueError):
            raw_delay = None
        invalid = raw_delay is not None and (not math.isfinite(raw_delay) or raw_delay < 0 or raw_delay > 1800)
        if pred_dt and actual_dt:
            delta = (actual_dt - pred_dt).total_seconds()
            if delta < 0 or delta > 1800:
                invalid = True
        if invalid:
            for key in ("actual_delay_sec", "timing_learning_delay_sec", "actual_delay", "delay_seconds"):
                if key in record:
                    record[key] = None
            record["actual_delay_source"] = "invalid_legacy_timestamp"
            changed["feedback"] += 1

    for record in experience_memory if isinstance(experience_memory, list) else []:
        if not isinstance(record, dict):
            continue
        pred_dt = parse_stamp(record.get("timestamp") or record.get("pred_time") or record.get("predicted_datetime"))
        actual_dt = parse_stamp(record.get("actual_time") or record.get("actual_datetime"))
        raw_delay = record.get("actual_delay_sec", record.get("actual_delay", record.get("delay_seconds")))
        try:
            raw_delay = float(raw_delay) if raw_delay is not None else None
        except (TypeError, ValueError):
            raw_delay = None
        invalid = raw_delay is not None and (not math.isfinite(raw_delay) or raw_delay < 0 or raw_delay > 1800)
        if pred_dt and actual_dt:
            delta = (actual_dt - pred_dt).total_seconds()
            if delta < 0 or delta > 1800:
                invalid = True
        if invalid:
            for key in ("actual_delay_sec", "actual_delay", "delay_seconds", "actual_delay_min", "timing_error", "delay_error_seconds", "timing_bucket", "timing_correct"):
                if key in record:
                    record[key] = None
            changed["experience"] += 1

    for record in feedback_samples if isinstance(feedback_samples, list) else []:
        if not isinstance(record, dict):
            continue
        pred_dt = parse_stamp(record.get("prediction_datetime"))
        actual_dt = parse_stamp(record.get("actual_datetime"))
        raw_delay = record.get("delay_seconds")
        try:
            raw_delay = float(raw_delay) if raw_delay is not None else None
        except (TypeError, ValueError):
            raw_delay = None
        invalid = raw_delay is not None and (not math.isfinite(raw_delay) or raw_delay < 0 or raw_delay > 1800)
        if pred_dt and actual_dt:
            delta = (actual_dt - pred_dt).total_seconds()
            if delta < 0 or delta > 1800:
                invalid = True
        if invalid:
            record["delay_seconds"] = None
            record["delay_minutes"] = None
            changed["samples"] += 1

    if any(changed.values()):
        logging.warning("[TIMING SANITIZER] mesures temporelles invalides neutralisées (sans supprimer les feedbacks): %s", changed)
        # Persist only within this bot's active data directory. Raw user uploads
        # and source files are not touched.
        try:
            if changed["feedback"]:
                save_file("feedback.pkl", feedback_history)
            if changed["experience"]:
                save_file("experience_memory.pkl", experience_memory)
            if changed["samples"]:
                save_file("feedback_samples.pkl", feedback_samples)
        except Exception as exc:
            logging.warning("[TIMING SANITIZER] sauvegarde différée: %s", exc)
    return changed


# ══════════════════════════════════════════════
# LOAD ALL
# ══════════════════════════════════════════════

def load_all():

    global pattern_memory
    global user_history
    global feedback_history
    global pending_predictions

    global feedback_samples
    global experience_memory
    global training_dataset
    global history

    global delay_memory
    global delay_engine
    global hour_delay_memory
    global delay_by_mode

    global realtime_delay_log
    global high_cote_observations

    global advanced_pattern_memory
    global pattern_stats
    global SEQ_WEIGHTS

    global AI_STATE
    global meta_model_stats
    global DELAY_STATS


    # ══════════════════════════════════════
    # LOAD FILES
    # ══════════════════════════════════════

    pattern_memory = load_file(
        "pattern_memory.pkl",
        pattern_memory
    )

    user_history = load_file(
        "user_history.pkl",
        user_history
    )

    feedback_history = normalize_feedback_history(
        load_file(
            "feedback.pkl",
            feedback_history
        )
    )

    pending_predictions = load_file(
        "pending_predictions.pkl",
        load_file("pending.pkl", pending_predictions)
    )
    pending_predictions = _normalize_pending_predictions(
        pending_predictions
    )

    feedback_samples = load_file(
        "feedback_samples.pkl",
        feedback_samples
    )

    experience_memory = load_file(
        "experience_memory.pkl",
        experience_memory
    )

    training_dataset = load_file(
        "training_dataset.pkl",
        training_dataset
    )

    history = load_file(
        "history.pkl",
        history
    )

    delay_memory = load_file(
        "delay_memory.pkl",
        delay_memory
    )

    delay_engine = load_file(
        "delay_engine.pkl",
        delay_engine
    )

    hour_delay_memory = load_file(
        "hour_delay_memory.pkl",
        hour_delay_memory
    )

    delay_by_mode = load_file(
        "delay_by_mode.pkl",
        delay_by_mode
    )

    realtime_delay_log = load_file(
        "realtime_delay_log.pkl",
        realtime_delay_log
    )

    high_cote_observations = load_file(
        "high_cote_observations.pkl",
        high_cote_observations
    )

    advanced_pattern_memory = load_file(
        "advanced_pattern_memory.pkl",
        advanced_pattern_memory
    )

    pattern_stats = load_file(
        "pattern_stats.pkl",
        pattern_stats
    )

    # Neutraliser les anciennes durées aberrantes avant toute reconstruction.
    # Les résultats directionnels et les cotes réelles restent conservés.
    try:
        sanitize_legacy_timing_records()
    except Exception as exc:
        logging.warning("[TIMING SANITIZER] %s", exc)

    # ============================================================
    # RECONSTRUCTION MÉMOIRE TEMPORELLE PATTERN v6.0
    # ============================================================
    try:
        rebuild_pattern_timing_memory()
        rebuild_sequence_feature_memory()

        save_file(
            "pattern_stats.pkl",
            pattern_stats
        )

        logging.info(
            "🧠 Mémoire PATTERN → DÉLAI reconstruite et sauvegardée."
        )

    except Exception as e:
        logging.warning(
            f"⚠️ Reconstruction timing pattern impossible : {e}"
        )


    # ══════════════════════════════════════
    # TYPE SAFETY
    # ══════════════════════════════════════

    if not isinstance(
        feedback_samples,
        list
    ):
        feedback_samples = []

    if not isinstance(
        experience_memory,
        list
    ):
        experience_memory = []

    if not isinstance(
        training_dataset,
        list
    ):
        training_dataset = []

    if not isinstance(
        history,
        list
    ):
        history = []

    if not isinstance(
        high_cote_observations,
        list
    ):
        high_cote_observations = []


    # ══════════════════════════════════════
    # PATTERN MEMORY
    # ══════════════════════════════════════

    if not isinstance(
        pattern_memory,
        dict
    ):

        pattern_memory = {
            "gains": {},
            "pertes": {},
        }


    for key in (
        "gains",
        "pertes"
    ):

        if not isinstance(
            pattern_memory.get(key),
            dict
        ):

            pattern_memory[key] = {}


        for L in (
            5,
            10,
            15
        ):

            value = pattern_memory[
                key
            ].get(
                L,
                []
            )

            if not isinstance(
                value,
                list
            ):

                value = []

            pattern_memory[
                key
            ][L] = value


    # ══════════════════════════════════════
    # LOAD AI STATE
    # ══════════════════════════════════════

    try:

        if os.path.exists(
            AI_STATE_FILE
        ):

            with open(
                AI_STATE_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                state = json.load(f)


            loaded_weights = state.get(
                "SEQ_WEIGHTS",
                {}
            )

            if isinstance(
                loaded_weights,
                dict
            ):

                for key, value in (
                    loaded_weights.items()
                ):

                    try:

                        key_int = int(key)

                        value_float = safe_float(
                            value,
                            SEQ_WEIGHTS.get(
                                key_int,
                                0.0
                            )
                        )

                        if (
                            key_int in
                            (5, 10, 15)
                        ):

                            SEQ_WEIGHTS[
                                key_int
                            ] = clamp(
                                value_float,
                                0.0,
                                1.0
                            )

                    except Exception:
                        continue


            stored_primary_id = state.get(
                "primary_script_id"
            )
            if (
                stored_primary_id
                and stored_primary_id != PRIMARY_SCRIPT_ID
                and stored_primary_id not in LEGACY_SCRIPT_IDS
            ):
                logging.warning(
                    "[PRIMARY] état IA d'un autre script ignoré : %s",
                    stored_primary_id
                )
                AI_STATE = {}
            else:
                loaded_ai_state = state.get(
                    "AI_STATE",
                    {}
                )

                if isinstance(loaded_ai_state, dict):
                    AI_STATE = loaded_ai_state
                    if stored_primary_id in LEGACY_SCRIPT_IDS:
                        logging.info(
                            "[PRIMARY] migration de l'état IA historique vers Rocket Queen V7."
                        )
                    AI_STATE["primary_script_id"] = PRIMARY_SCRIPT_ID
                    AI_STATE["primary_script_version"] = PRIMARY_SCRIPT_VERSION
                else:
                    AI_STATE = {}


    except Exception as e:

        logging.warning(
            f"load_ai_state: {e}"
        )

        AI_STATE = {}


    # ══════════════════════════════════════
    # FEATURE VERSION
    # ══════════════════════════════════════

    stored_version = safe_float(
        AI_STATE.get(
            "feature_version",
            0
        ),
        0
    )


    if int(stored_version) != FEATURE_VERSION:
        # Never silently discard historical training rows. Archive incompatible
        # rows unchanged, while retaining any rows already compatible with the
        # current 12-feature schema for safe training.
        compatible_rows = []
        incompatible_rows = []
        for item in training_dataset if isinstance(training_dataset, list) else []:
            if not isinstance(item, dict):
                incompatible_rows.append(item)
                continue
            features = item.get("features", [])
            label = safe_float(item.get("label"), -1.0)
            compatible = (item.get("feature_version") == FEATURE_VERSION
                          and isinstance(features, (list, tuple)) and len(features) == 12
                          and label in (0.0, 1.0))
            if compatible:
                copied = dict(item)
                copied["features"] = list(features)
                compatible_rows.append(copied)
            else:
                incompatible_rows.append(item)

        if incompatible_rows:
            archive_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive_name = f"archives/training_dataset_legacy_feature_v{int(stored_version)}_{archive_stamp}.pkl"
            save_file(archive_name, incompatible_rows)
            logging.warning(
                "[FEATURE_VERSION] %s lignes incompatibles archivées dans %s; aucune ligne historique supprimée.",
                len(incompatible_rows), archive_name
            )

        training_dataset = compatible_rows
        logging.info(
            "[FEATURE_VERSION] stored=%s current=%s — lignes compatibles conservées=%s, anciennes lignes archivées=%s",
            stored_version, FEATURE_VERSION, len(training_dataset), len(incompatible_rows)
        )
        AI_STATE["feature_version"] = FEATURE_VERSION


    # ══════════════════════════════════════
    # MODULE 2
    # ══════════════════════════════════════

    if not isinstance(
        AI_STATE.get(
            "pattern_memory"
        ),
        dict
    ):

        # create_context_memory() appartient
        # aux parties suivantes.
        # On garde un fallback sûr.

        AI_STATE[
            "pattern_memory"
        ] = {}


    # ══════════════════════════════════════
    # MODULE 4 — MODEL PERFORMANCE
    # ══════════════════════════════════════

    if not isinstance(
        AI_STATE.get(
            "model_performance"
        ),
        dict
    ):

        AI_STATE[
            "model_performance"
        ] = {}


    # ══════════════════════════════════════
    # MODULE 3 — GLOBAL COUNTERS
    # ══════════════════════════════════════

    AI_STATE.setdefault(
        "total_gains",
        0
    )

    AI_STATE.setdefault(
        "total_losses",
        0
    )


    # ══════════════════════════════════════
    # MODULES 7/8/9
    # ══════════════════════════════════════

    saved_meta = AI_STATE.get(
        "meta_model_stats"
    )

    if isinstance(
        saved_meta,
        dict
    ):

        meta_model_stats = saved_meta

    else:

        meta_model_stats = {}


    AI_STATE[
        "meta_model_stats"
    ] = meta_model_stats


    # ══════════════════════════════════════
    # DELAY STATS
    # ══════════════════════════════════════

    saved_delay_stats = AI_STATE.get(
        "DELAY_STATS"
    )


    if isinstance(
        saved_delay_stats,
        dict
    ):

        for delay in SUPPORTED_DELAYS:

            saved_bucket = (
                saved_delay_stats.get(
                    str(delay)
                )
            )

            if isinstance(
                saved_bucket,
                dict
            ):

                DELAY_STATS[
                    delay
                ].update(
                    saved_bucket
                )


    normalize_delay_stats()


    AI_STATE[
        "DELAY_STATS"
    ] = {
        str(k): v
        for k, v in DELAY_STATS.items()
    }


    # ══════════════════════════════════════
    # TIMING MEMORY
    # ══════════════════════════════════════

    AI_STATE.setdefault(
        "timing_memory",
        {}
    )

    normalize_timing_memory()

    # Rebuild delay scores from the canonical feedback ledger. Persisted
    # counters may come from older versions that credited one observation to
    # both the selected and observed delay windows.
    rebuild_delay_stats_from_feedback()


    # ══════════════════════════════════════
    # MIGRATION
    # ══════════════════════════════════════

    save_file(
        "feedback.pkl",
        feedback_history
    )


    # ══════════════════════════════════════
    # RETRAINING
    # ══════════════════════════════════════

    if len(
        training_dataset
    ) >= 20:

        _train_model_on_dataset(
            training_dataset
        )


    logging.info(
        "✅ Loaded — "
        f"feedbacks={len(feedback_history)} | "
        f"experiences={len(experience_memory)} | "
        f"dataset={len(training_dataset)}"
    )


# ══════════════════════════════════════════════
# SAVE ALL
# ══════════════════════════════════════════════

def save_all():

    save_file(
        "pattern_memory.pkl",
        pattern_memory
    )

    save_file(
        "advanced_pattern_memory.pkl",
        advanced_pattern_memory
    )

    save_file(
        "pattern_stats.pkl",
        pattern_stats
    )

    save_file(
        "training_dataset.pkl",
        training_dataset
    )

    save_file(
        "feedback.pkl",
        feedback_history
    )

    save_file(
        "feedback_samples.pkl",
        feedback_samples
    )

    save_file(
        "history.pkl",
        history
    )

    save_file(
        "pending_predictions.pkl",
        pending_predictions
    )
    # Keep the old filename in sync for rollback compatibility.
    save_file(
        "pending.pkl",
        pending_predictions
    )

    save_file(
        "user_history.pkl",
        user_history
    )

    save_file(
        "experience_memory.pkl",
        experience_memory
    )

    save_file(
        "delay_memory.pkl",
        delay_memory
    )

    save_file(
        "delay_engine.pkl",
        delay_engine
    )

    save_file(
        "hour_delay_memory.pkl",
        hour_delay_memory
    )

    save_file(
        "delay_by_mode.pkl",
        delay_by_mode
    )

    save_file(
        "realtime_delay_log.pkl",
        realtime_delay_log
    )

    save_file(
        "high_cote_observations.pkl",
        high_cote_observations
    )


    # ══════════════════════════════════════
    # AI STATE
    # ══════════════════════════════════════

    try:

        AI_STATE[
            "DELAY_STATS"
        ] = {
            str(k): v
            for k, v in DELAY_STATS.items()
        }


        AI_STATE[
            "meta_model_stats"
        ] = meta_model_stats


        AI_STATE[
            "feature_version"
        ] = FEATURE_VERSION

        AI_STATE[
            "primary_script_id"
        ] = PRIMARY_SCRIPT_ID

        AI_STATE[
            "primary_script_version"
        ] = PRIMARY_SCRIPT_VERSION


        state = {

            "primary_script_id": PRIMARY_SCRIPT_ID,
            "primary_script_version": PRIMARY_SCRIPT_VERSION,

            "SEQ_WEIGHTS": {
                str(k): float(v)
                for k, v in SEQ_WEIGHTS.items()
            },

            "AI_STATE": AI_STATE,

        }


        temp_state = (
            f"{AI_STATE_FILE}.tmp"
        )


        with open(
            temp_state,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                state,
                f,
                indent=2,
                ensure_ascii=False
            )


        os.replace(
            temp_state,
            AI_STATE_FILE
        )


    except Exception as e:

        logging.warning(
            f"save_ai_state: {e}"
        )


# ══════════════════════════════════════════════
# MEMORY PRUNING
# ══════════════════════════════════════════════

def _prune_memory():

    global feedback_samples
    global experience_memory
    global high_cote_observations


    if len(
        feedback_samples
    ) > MAX_FEEDBACK_SAMPLES:

        feedback_samples = (
            feedback_samples[
                -MAX_FEEDBACK_SAMPLES:
            ]
        )


    if len(
        experience_memory
    ) > MAX_EXPERIENCE_MEMORY:

        experience_memory = (
            experience_memory[
                -MAX_EXPERIENCE_MEMORY:
            ]
        )


    if len(
        high_cote_observations
    ) > MAX_HIGH_COTE_OBSERVATIONS:

        high_cote_observations = (
            high_cote_observations[
                -MAX_HIGH_COTE_OBSERVATIONS:
            ]
        )


# ══════════════════════════════════════════════
# HIGH COTE — HOUR SIGNAL
# ══════════════════════════════════════════════

def get_high_cote_hour_signal(
    hour: int
) -> dict:

    matches = []

    for observation in (
        high_cote_observations
    ):

        if not isinstance(
            observation,
            dict
        ):
            continue

        try:

            if int(
                observation.get(
                    "hour"
                )
            ) != int(hour):

                continue


            cote = safe_float(
                observation.get(
                    "cote",
                    0.0
                )
            )


            if cote > 10.0:

                matches.append(
                    cote
                )

        except (
            TypeError,
            ValueError
        ):

            continue


    count = len(
        matches
    )


    pressure = clamp(
        count / 5.0,
        0.0,
        1.0
    )


    return {

        "hour": int(hour),

        "observations": count,

        "pressure": round(
            pressure,
            4
        ),

        "average_cote": (
            round(
                sum(matches) / count,
                2
            )
            if count
            else None
        ),

        "max_cote": (
            round(
                max(matches),
                2
            )
            if matches
            else None
        ),

    }


# ══════════════════════════════════════════════
# HIGH COTE — TIME SIGNAL
# ══════════════════════════════════════════════

def get_high_cote_time_signal(
    hour: int,
    minute: int,
    cotes: list = None,
    window_minutes: int = 25,
) -> dict:

    """
    Estime un signal historique >10x.

    IMPORTANT :
    Ce signal est uniquement une composante.

    Il ne doit jamais être interprété comme :
        "une grosse cote va forcément arriver".

    Il doit être fusionné ensuite avec :
        - patterns,
        - mémoire,
        - timing,
        - feedback,
        - modèles.
    """

    now_total = (
        int(hour) * 60 +
        int(minute)
    )


    # ══════════════════════════════════════
    # 1 — OBSERVATIONS TEMPORELLES
    # ══════════════════════════════════════

    nearby = []


    for obs in (
        high_cote_observations
    ):

        if not isinstance(
            obs,
            dict
        ):
            continue


        try:

            obs_hour = int(
                obs.get(
                    "hour",
                    -1
                )
            )

            obs_minute = int(
                obs.get(
                    "minute",
                    0
                )
            )


            obs_total = (
                obs_hour * 60 +
                obs_minute
            )


            dist = abs(
                obs_total -
                now_total
            )


            # Passage minuit
            dist = min(
                dist,
                1440 - dist
            )


            if dist <= window_minutes:

                cote_val = safe_float(
                    obs.get(
                        "cote",
                        0.0
                    )
                )


                if cote_val > 10.0:

                    nearby.append({

                        "dist": dist,

                        "cote": cote_val,

                        "recorded_at": obs.get(
                            "recorded_at",
                            ""
                        ),

                        "hour": obs_hour,

                        "minute": obs_minute,

                    })


        except (
            TypeError,
            ValueError
        ):

            continue


    count = len(
        nearby
    )


    if count == 0:

        return {

            "probability": 0.0,

            "observations": 0,

            "nearby_obs": [],

            "avg_cote": None,

            "max_cote": None,

            "closest_minutes": None,

            "label": "none",

        }


    # ══════════════════════════════════════
    # 2 — PROXIMITÉ TEMPORELLE
    # ══════════════════════════════════════

    sigma = max(
        window_minutes / 2.0,
        1.0
    )


    weights = [

        math.exp(
            -0.5 *
            (
                o["dist"] /
                sigma
            ) ** 2
        )

        for o in nearby
    ]


    total_weight = sum(
        weights
    )


    baseline = max(
        1.0,
        window_minutes / 30.0
    )


    proximity_prob = clamp(
        total_weight /
        (
            baseline * 2.0
        ),
        0.0,
        1.0
    )


    # ══════════════════════════════════════
    # 3 — FRAÎCHEUR
    # ══════════════════════════════════════

    now_ts = time.time()

    recency_scores = []


    for obs in nearby:

        try:

            recorded_at = (
                obs.get(
                    "recorded_at",
                    ""
                )
            )

            parsed = (
                datetime.datetime
                .fromisoformat(
                    recorded_at
                )
            )


            # Si le timestamp est naïf,
            # on le considère dans le fuseau local.
            if parsed.tzinfo is None:

                parsed_ts = (
                    parsed.timestamp()
                )

            else:

                parsed_ts = (
                    parsed.timestamp()
                )


            days_old = max(
                0.0,
                (
                    now_ts -
                    parsed_ts
                ) / 86400.0
            )


            recency_scores.append(
                math.exp(
                    -0.1 *
                    days_old
                )
            )


        except Exception:

            # Une observation sans timestamp
            # ne doit pas être considérée
            # comme parfaitement fraîche.
            recency_scores.append(
                0.5
            )


    recency_prob = clamp(
        sum(recency_scores) /
        max(
            len(recency_scores),
            1
        ),
        0.0,
        1.0
    )


    # ══════════════════════════════════════
    # 4 — PATTERN ACTUEL
    # ══════════════════════════════════════

    pattern_prob = 0.50


    if cotes and len(cotes) >= 5:

        last5 = []

        for c in cotes[-5:]:

            value = safe_float(
                c,
                None
            )

            if value is not None:

                last5.append(
                    value
                )


        if last5:

            avg5 = (
                sum(last5) /
                len(last5)
            )

            low_count = sum(
                1
                for c in last5
                if c < 2.0
            )


            if (
                low_count >= 4
                and avg5 < 2.0
            ):

                pattern_prob = 0.85

            elif (
                low_count >= 3
                and avg5 < 2.5
            ):

                pattern_prob = 0.72

            elif low_count >= 2:

                pattern_prob = 0.58

            elif avg5 > 6.0:

                pattern_prob = 0.30

            else:

                pattern_prob = 0.50


    # ══════════════════════════════════════
    # 5 — COMBINAISON
    # ══════════════════════════════════════

    probability = clamp(

        0.50 *
        proximity_prob

        +

        0.25 *
        recency_prob

        +

        0.25 *
        pattern_prob,

        0.0,
        1.0
    )


    cotes_list = [
        o["cote"]
        for o in nearby
    ]

    dists = [
        o["dist"]
        for o in nearby
    ]


    label = (

        "high"
        if probability >= 0.60

        else

        "medium"
        if probability >= 0.35

        else

        "low"
    )


    return {

        "probability": round(
            probability,
            3
        ),

        "observations": count,

        "nearby_obs": nearby,

        "avg_cote": round(
            sum(cotes_list) /
            count,
            2
        ),

        "max_cote": round(
            max(cotes_list),
            2
        ),

        "closest_minutes": round(
            min(dists),
            1
        ),

        "label": label,

    }


# ══════════════════════════════════════════════
# SPLIT ANALYSIS COTES
# ══════════════════════════════════════════════

def split_analysis_cotes(
    cotes: list
) -> tuple:

    """
    Sépare les cotes normales des événements >10x.

    Les >10x ne sont PAS supprimées de la mémoire :
    elles sont simplement séparées afin de ne pas
    déformer certaines statistiques normales.
    """

    normal = []
    high = []


    for cote in cotes or []:

        value = safe_float(
            cote,
            None
        )

        if value is None:
            continue


        if value > 10.0:

            high.append(
                round(
                    value,
                    3
                )
            )

        else:

            normal.append(
                round(
                    value,
                    3
                )
            )


    return (
        normal,
        high
    )


# ══════════════════════════════════════════════
# MODULE 1 — INTELLIGENT FEEDBACK
# ══════════════════════════════════════════════

def parse_real_datetime(
    raw: str,
    prediction_dt: datetime.datetime
) -> Optional[datetime.datetime]:
    """Parse l'heure saisie sans inventer un passage au jour suivant.

    Les choix de timing couvrent les fenêtres de 1, 2 ou 3 minutes. On accepte une
    observation tardive jusqu'à 30 minutes afin de mesurer une erreur réelle, mais
    une heure seule antérieure à la prédiction reste ambiguë : aucun jour suivant
    n'est inventé automatiquement. Pour traverser minuit, fournir une date complète.
    """
    if not raw:
        return None

    raw = str(raw).strip()
    # Si une date complète est fournie, elle fait foi.
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ):
        try:
            result = datetime.datetime.strptime(raw, fmt)
            delta = (result - prediction_dt).total_seconds()
            if delta < 0 or delta > 1800:
                return None
            return result
        except ValueError:
            continue

    # Sans date, on conserve le jour de la prédiction. Ne jamais ajouter
    # automatiquement 24 h : une saisie proche de l'heure de prédiction peut
    # simplement manquer de précision à la minute.
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.datetime.strptime(raw, fmt)
            result = parsed.replace(
                year=prediction_dt.year,
                month=prediction_dt.month,
                day=prediction_dt.day,
            )
            delta = (result - prediction_dt).total_seconds()
            if delta < 0:
                return None
            if delta > 1800:
                return None
            return result
        except ValueError:
            continue
    return None


# ══════════════════════════════════════════════
# CREATE FEEDBACK SAMPLE
# ══════════════════════════════════════════════

def create_feedback_sample(
    chat_id: int,
    prediction: dict,
    status: str,
    actual_datetime: Optional[
        datetime.datetime
    ],
    max_cote_reached: float,
    turns_between: Optional[int] = None,
    low_cote_losses: int = 0,
) -> FeedbackSample:

    result = (
        prediction.get(
            "result",
            {}
        )
    )


    cotes = (
        prediction.get(
            "cotes",
            []
        )
    )


    pred_iso = (
        prediction.get(
            "pred_time"
        )
    )


    try:

        prediction_dt = (
            datetime.datetime
            .fromisoformat(
                pred_iso
            )
        )

    except Exception:

        prediction_dt = (
            datetime.datetime.now()
        )


    # ══════════════════════════════════════
    # NORMALIZE CURRENT COTES
    # ══════════════════════════════════════

    last = clean_numeric_sequence(
        cotes
    )[-ANALYSE_LAST:]


    m, s = mean_std(
        last
    )

    vol = volatility(
        last
    )

    low = consecutive_low(
        last
    )

    peak = (
        compute_peak_interval(
            last
        )
        or 0.0
    )


    # ══════════════════════════════════════
    # REAL OBSERVED DELAY
    # ══════════════════════════════════════

    delay_seconds = None


    if actual_datetime is not None:
        # Ne pas transformer automatiquement une heure antérieure en lendemain :
        # cela créait des délais proches de 24 h à partir d'une simple saisie HH:MM.
        measured_delay = (actual_datetime - prediction_dt).total_seconds()
        if 0 <= measured_delay <= 1800:
            delay_seconds = measured_delay
        else:
            delay_seconds = None
            actual_datetime = None


    delay_minutes = (

        delay_seconds / 60.0

        if delay_seconds is not None

        else None
    )


    # ══════════════════════════════════════
    # CREATE SAMPLE
    # ══════════════════════════════════════

    sample_id = hashlib.sha256(

        f"{chat_id}_"
        f"{prediction_dt.isoformat()}_"
        f"{time.time_ns()}_"
        f"{uuid.uuid4()}"

        .encode(
            "utf-8"
        )

    ).hexdigest()[:16]


    return FeedbackSample(

        sample_id=sample_id,

        chat_id=chat_id,

        mode=prediction.get(
            "mode",
            "unknown"
        ),

        predicted_target=safe_float(
            result.get(
                "target",
                0.0
            )
        ),

        predicted_assurance=safe_float(
            result.get(
                "assurance",
                0.0
            )
        ),

        predicted_reliability=safe_float(
            result.get(
                "fiabilite",
                0.0
            )
        ),

        predicted_score=clamp(
            safe_float(
                result.get(
                    "score",
                    0.0
                )
            ),
            0.0,
            1.0
        ),

        prediction_datetime=(
            prediction_dt.isoformat()
        ),

        predicted_hour=prediction_dt.hour,

        predicted_minute=prediction_dt.minute,

        predicted_second=prediction_dt.second,

        actual_datetime=(
            actual_datetime.isoformat()
            if actual_datetime
            else None
        ),

        actual_hour=(
            actual_datetime.hour
            if actual_datetime
            else None
        ),

        actual_minute=(
            actual_datetime.minute
            if actual_datetime
            else None
        ),

        actual_second=(
            actual_datetime.second
            if actual_datetime
            else None
        ),

        delay_seconds=delay_seconds,

        delay_minutes=delay_minutes,

        turns_between=turns_between,

        low_cote_losses=max(
            0,
            int(
                low_cote_losses or 0
            )
        ),

        status=str(
            status or "unknown"
        ),

        max_cote_reached=max(
            0.0,
            safe_float(
                max_cote_reached
            )
        ),

        mean_cote=float(m),

        std_cote=float(s),

        volatility=float(vol),

        consecutive_low=int(low),

        peak_interval=float(peak),

        sequence=list(last),

        created_at=(
            datetime.datetime.now()
            .isoformat()
        ),

        prediction_id=prediction.get("prediction_id"),

    )


# ══════════════════════════════════════════════
# REGISTER FEEDBACK
# ══════════════════════════════════════════════

def register_feedback_sample(
    sample: FeedbackSample
):

    if not isinstance(
        sample,
        FeedbackSample
    ):
        return


    try:
        sample_payload = asdict(sample)
    except Exception:
        # Backward compatibility for legacy pickled dataclass instances that
        # predate newly added optional fields.
        sample_payload = {
            name: getattr(sample, name, None)
            for name in FeedbackSample.__dataclass_fields__
        }
    sample_payload["prediction_id"] = getattr(sample, "prediction_id", None)
    feedback_samples.append(sample_payload)


    if len(
        feedback_samples
    ) > MAX_FEEDBACK_SAMPLES:

        del feedback_samples[
            :-MAX_FEEDBACK_SAMPLES
        ]


# ══════════════════════════════════════════════
# PREDICTION QUALITY
# ══════════════════════════════════════════════

def compute_prediction_quality(
    sample: FeedbackSample
) -> float:

    if not isinstance(
        sample,
        FeedbackSample
    ):

        return 0.0


    quality = 0.0


    # ══════════════════════════════════════
    # TARGET REACHED
    # ══════════════════════════════════════

    if (
        sample.max_cote_reached
        >=
        sample.predicted_target
        and
        sample.predicted_target > 0
    ):

        quality += 0.50


    # ══════════════════════════════════════
    # TARGET ERROR
    # ══════════════════════════════════════

    if sample.predicted_target > 0:

        error_ratio = (

            abs(
                sample.max_cote_reached
                -
                sample.predicted_target
            )

            /

            max(
                sample.predicted_target,
                0.01
            )

        )


        quality += (

            0.25 *

            max(
                0.0,
                1.0 -
                error_ratio
            )

        )


    # ══════════════════════════════════════
    # TIMING QUALITY
    # ══════════════════════════════════════

    if (
        sample.delay_seconds
        is not None
    ):

        timing_component = math.exp(

            -abs(
                sample.delay_seconds
            )
            /
            120.0

        )


        quality += (
            0.25 *
            clamp(
                timing_component,
                0.0,
                1.0
            )
        )


    return clamp(
        quality,
        0.0,
        1.0
    )


# ══════════════════════════════════════════════
# BUILD TRAINING SAMPLE
# ══════════════════════════════════════════════

def build_training_sample_from_feedback(
    sample: FeedbackSample
) -> dict:

    quality = (
        compute_prediction_quality(
            sample
        )
    )


    # Direction uniquement : early/late ne sont jamais transformés en perte.
    label = 1 if str(sample.status).lower() == "gain" else 0


    features = build_prediction_features(

        consecutive_low_value=(
            sample.consecutive_low
        ),

        volatility_value=(
            sample.volatility
        ),

        std_value=(
            sample.std_cote
        ),

        mean_value=(
            sample.mean_cote
        ),

        predicted_score=0.0,
        predicted_target=0.0,

        predicted_hour=(
            sample.predicted_hour
        ),

        predicted_minute=(
            sample.predicted_minute
        ),

        peak_interval=(
            sample.peak_interval
        ),

        sequence_length=len(
            sample.sequence
        ),

        pattern_gain=0.0,

        pattern_loss=0.0,

    )


    return {

        "features": features,

        "label": label,

        "quality": quality,

        # Métadonnée post-prédiction.
        # Elle ne doit PAS devenir une feature
        # utilisée avant le résultat.

        "low_cote_losses": int(
            getattr(
                sample,
                "low_cote_losses",
                0
            )
            or 0
        ),

        "sample_id": (
            sample.sample_id
        ),

        "feature_version": (
            FEATURE_VERSION
        ),

        "created_at": (
            sample.created_at
        ),

    }

# ══════════════════════════════════════════════════════════════════════
# MODULE 2 — SMART MEMORY / EXPERIENCE MEMORY
# ══════════════════════════════════════════════════════════════════════

def create_experience_memory(
    cotes,
    mode,
    ai_score,
    prediction_result,
    markov_confidence=0.0,
    neural_score_val=0.5,
    neural_conf_val=0.0,
) -> Optional[dict]:
    """
    Crée une expérience au moment de la prédiction.

    IMPORTANT :
    Cette expérience est créée avec outcome='pending'.
    Elle sera complétée lors du feedback.

    La mémoire conserve maintenant séparément :
      - la qualité du signal
      - la cible
      - le délai prédit
      - le contexte temporel
      - les preuves utilisées
    """

    if not cotes:
        return None

    last = cotes[-ANALYSE_LAST:]

    m, s = mean_std(last)
    low = consecutive_low(last)
    vol = volatility(last)
    stab = stability_score(last)
    ent = entropy_score(last)
    bayes = bayes_probability(last)
    markov = weighted_markov_next(last)

    pat = detect_weighted_patterns(
        last,
        pattern_memory,
        SEQ_WEIGHTS
    )

    now = datetime.datetime.now()

    # --------------------------------------------------------------
    # Sécurisation du résultat de prédiction
    # --------------------------------------------------------------

    prediction_result = prediction_result or {}

    target = float(
        prediction_result.get("target", 0.0) or 0.0
    )

    predicted_delay = float(
        prediction_result.get("delay_sec", 0.0) or 0.0
    )

    optimal_time = prediction_result.get(
        "optimal_time",
        ""
    )

    fenetre = prediction_result.get(
        "fenetre",
        ("", "")
    )

    if not isinstance(fenetre, (list, tuple)) or len(fenetre) < 2:
        fenetre = ("", "")

    # --------------------------------------------------------------
    # Création de l'expérience
    # --------------------------------------------------------------

    exp = ExperienceMemory(
        experience_id=uuid.uuid4().hex,
        timestamp=now.isoformat(),

        hour=now.hour,
        minute=now.minute,

        mode=mode,

        cotes=list(last),

        mean=round(m, 6),
        std=round(s, 6),

        low_streak=low,
        volatility=round(vol, 6),
        stability=round(stab, 6),
        entropy=round(ent, 6),

        bayes_probability=round(bayes, 6),

        markov_prediction=round(markov, 6),
        markov_confidence=round(markov_confidence, 6),

        pattern_score=round(
            pat.get("net", 0.0),
            6
        ),

        pattern_gain_score=round(
            pat.get("raw_gain", 0.0),
            6
        ),

        pattern_loss_score=round(
            pat.get("raw_loss", 0.0),
            6
        ),

        neural_score=round(
            neural_score_val,
            6
        ),

        neural_confidence=round(
            neural_conf_val,
            6
        ),

        ai_score=round(
            ai_score,
            6
        ),

        target=target,

        predicted_delay=predicted_delay,

        predicted_time=optimal_time,

        window_start=fenetre[0],
        window_end=fenetre[1],

        created_at=now.isoformat(),
        updated_at=now.isoformat(),

        low_cote_losses=0,
    )

    result = asdict(exp)

    # --------------------------------------------------------------
    # État initial
    # --------------------------------------------------------------

    result["outcome"] = "pending"
    result["status"] = "pending"

    result["actual_delay"] = None
    result["actual_delay_sec"] = None
    result["delay_seconds"] = None
    result["delay_sec"] = predicted_delay
    result["predicted_delay_sec"] = predicted_delay

    result["timing_error"] = None
    result["target_error"] = None

    result["real_time"] = None
    result["actual_time"] = None

    result["real_max_cote"] = None
    result["actual_max_cote"] = None

    result["reliability"] = EXP_RELIABILITY_DEFAULT
    result["verified_count"] = 0

    # --------------------------------------------------------------
    # Trace complète de décision
    # --------------------------------------------------------------

    result["timing_context_key"] = prediction_result.get(
        "timing_context_key"
    )

    result["delay_scores"] = dict(
        prediction_result.get("delay_scores", {}) or {}
    )

    result["timing_features"] = dict(
        prediction_result.get("timing_features", {}) or {}
    )

    result["timing_evidence"] = dict(
        prediction_result.get("timing_evidence", {}) or {}
    )

    result["model_agreement"] = prediction_result.get(
        "model_agreement"
    )

    result["evidence_confidence"] = prediction_result.get(
        "evidence_confidence"
    )

    result["direction_probability"] = prediction_result.get(
        "direction_probability"
    )

    # --------------------------------------------------------------
    # Champs dédiés à l'apprentissage temporel
    # --------------------------------------------------------------

    result["predicted_delay_min"] = round(
        predicted_delay / 60.0,
        4
    ) if predicted_delay > 0 else None

    result["actual_delay_min"] = None

    result["delay_error_seconds"] = None

    result["timing_correct"] = None

    result["timing_bucket"] = None

    return result


def store_experience(exp: dict) -> bool:
    """
    Stockage unique d'une expérience.

    Évite également de créer accidentellement plusieurs copies
    du même experience_id.
    """

    if not exp:
        return False

    exp_id = exp.get("experience_id")

    if exp_id:
        for existing in experience_memory:
            if (
                isinstance(existing, dict)
                and existing.get("experience_id") == exp_id
            ):
                return False

    experience_memory.append(exp)

    if len(experience_memory) > MAX_EXPERIENCE_MEMORY:
        excess = (
            len(experience_memory)
            - MAX_EXPERIENCE_MEMORY
        )

        if excess > 0:
            del experience_memory[:excess]

    return True


# ══════════════════════════════════════════════════════════════════════
# MODULE 3 — MEMORY-BASED LEARNING
# ══════════════════════════════════════════════════════════════════════

def experience_distance(cur, prev) -> float:
    """
    Distance entre deux expériences.

    La distance ne doit PAS utiliser directement le délai prédit,
    sinon deux expériences ayant reçu des délais différents seraient
    artificiellement considérées comme différentes.

    Le délai est appris APRÈS la comparaison.
    """

    if not cur or not prev:
        return float("inf")

    d = 0.0

    cur_mean = _safe_float(cur.get("mean", 0.0))
    prev_mean = _safe_float(prev.get("mean", 0.0))
    mean_scale = max(1.0, abs(cur_mean), abs(prev_mean))
    d += min(1.0, abs(cur_mean - prev_mean) / mean_scale) * 0.20

    cur_std = _safe_float(cur.get("std", 0.0))
    prev_std = _safe_float(prev.get("std", 0.0))
    std_scale = max(1.0, abs(cur_std), abs(prev_std))
    d += min(1.0, abs(cur_std - prev_std) / std_scale) * 0.15

    d += abs(
        _safe_float(cur.get("stability", 0.0))
        - _safe_float(prev.get("stability", 0.0))
    ) * 0.20

    d += abs(
        _safe_float(cur.get("entropy", 0.0))
        - _safe_float(prev.get("entropy", 0.0))
    ) * 0.15

    d += abs(
        _safe_float(cur.get("bayes_probability", cur.get("bayes", 0.0)))
        - _safe_float(prev.get("bayes_probability", prev.get("bayes", 0.0)))
    ) * 0.15

    d += abs(
        _safe_float(cur.get("ai_score", 0.0))
        - _safe_float(prev.get("ai_score", 0.0))
    ) * 0.10

    # Comparaison directe des séquences récentes : les cotes sont comparées
    # en espace logarithmique pour que 1.2x vs 1.5x compte plus que l'écart
    # brut entre de très grandes cotes. Les séquences sont alignées par la fin.
    cur_seq = cur.get("sequence", cur.get("cotes", [])) or []
    prev_seq = prev.get("sequence", prev.get("cotes", [])) or []
    try:
        cur_seq = [max(1.0, float(x)) for x in cur_seq[-15:]]
        prev_seq = [max(1.0, float(x)) for x in prev_seq[-15:]]
        n = min(len(cur_seq), len(prev_seq))
        if n:
            cs, ps = cur_seq[-n:], prev_seq[-n:]
            seq_diffs = [
                min(1.0, abs(math.log(a) - math.log(b)) / math.log(10.0))
                for a, b in zip(cs, ps)
            ]
            # Le segment le plus récent reçoit plus de poids.
            weights = [0.5 + (i + 1) / n for i in range(n)]
            seq_distance = sum(v*w for v, w in zip(seq_diffs, weights)) / sum(weights)
            d += 0.30 * seq_distance
    except Exception:
        pass

    # Le nombre de petites cotes consécutives fait partie du contexte.
    cur_low = _safe_float(cur.get("low_count", cur.get("low_streak", 0)))
    prev_low = _safe_float(prev.get("low_count", prev.get("low_streak", 0)))
    d += min(1.0, abs(cur_low - prev_low) / 10.0) * 0.10

    return float(d)


def experience_similarity(cur, prev) -> float:
    if not cur or not prev:
        return 0.0

    dist = experience_distance(cur, prev)

    if dist >= MEMORY_DISTANCE_THRESHOLD:
        return 0.0

    base_similarity = clamp(
        1.0 - (
            dist / MEMORY_DISTANCE_THRESHOLD
        ),
        0.0,
        1.0
    )

    # L'heure ne bloque jamais une expérience pertinente, mais permet de
    # favoriser légèrement les contextes horaires proches (heure circulaire).
    try:
        cur_minute = _safe_float(cur.get("hour", 0)) * 60 + _safe_float(cur.get("minute", 0))
        prev_minute = _safe_float(prev.get("hour", 0)) * 60 + _safe_float(prev.get("minute", 0))
        delta = abs(cur_minute - prev_minute) % 1440
        delta = min(delta, 1440 - delta)
        time_similarity = clamp(1.0 - delta / 360.0, 0.0, 1.0)
        base_similarity *= (0.90 + 0.10 * time_similarity)
    except Exception:
        pass

    return clamp(base_similarity, 0.0, 1.0)


def find_similar_experiences_v2(
    current_exp,
    limit=MAX_SIMILAR_EXPERIENCES
) -> list:

    if not current_exp or not experience_memory:
        return []

    current_mode = current_exp.get("mode")

    candidates = []

    for mem in experience_memory:

        if not isinstance(mem, dict):
            continue

        # ----------------------------------------------------------
        # Une expérience pending ne doit JAMAIS servir de feedback
        # ----------------------------------------------------------

        if mem.get("outcome") == "pending":
            continue

        if mem.get("status") == "pending":
            continue

        # ----------------------------------------------------------
        # On compare d'abord les expériences du même mode
        # ----------------------------------------------------------

        if mem.get("mode") != current_mode:
            continue

        sim = experience_similarity(
            current_exp,
            mem
        )

        if sim <= 0.0:
            continue

        candidates.append({
            "experience": mem,
            "similarity": sim,
        })

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return candidates[:limit]


def _weighted_average(values):
    if not values:
        return None

    total_weight = sum(
        float(weight)
        for _, weight in values
        if weight > 0
    )

    if total_weight <= 0:
        return None

    return sum(
        float(value) * float(weight)
        for value, weight in values
        if weight > 0
    ) / total_weight


def _delay_bucket(delay_seconds):
    """Retourne la fenêtre réelle observée : 1=[60,120), 2=[120,180), 3=[180,240).

    Une observation avant 60 s ou à partir de 240 s n'est pas artificiellement
    classée dans la fenêtre 1 ou 3 ; elle constitue un écart temporel hors fenêtre.
    """
    if delay_seconds is None:
        return None
    try:
        d = float(delay_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d < 60.0 or d >= 240.0:
        return None
    bucket = int(d // 60.0)
    return bucket if bucket in SUPPORTED_DELAYS else None


def compute_memory_learning(current_exp) -> dict:
    """
    Apprentissage historique complet.

    La mémoire produit maintenant séparément :

      - réussite historique
      - délai moyen réel
      - erreur moyenne de cible
      - pertes de parcours
      - distribution des délais
      - délai dominant
      - confiance temporelle

    Cela permet au moteur temporel des modules suivants de savoir
    ce qui s'est historiquement passé dans des situations similaires.
    """

    similar = find_similar_experiences_v2(
        current_exp
    )

    empty = {
        "available": False,
        "samples": len(similar),

        "success_rate": MEMORY_SUCCESS_NEUTRAL,

        "avg_delay": None,
        "avg_delay_min": None,

        "median_delay": None,
        "dominant_delay": None,

        "delay_distribution": {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
        },

        "timing_confidence": 0.0,
        "timing_samples": 0,
        "timing_weight": 0.0,

        "avg_target_error": None,
        "avg_low_cote_losses": None,

        "confidence": 0.0,

        "memory_score": MEMORY_SUCCESS_NEUTRAL,
        "influence": 0.0,
    }

    # Même avec peu d'expériences, on calcule un signal provisoire.
    # Sa confiance et son influence restent faibles tant que l'échantillon
    # est petit ; on ne jette donc pas les premiers feedbacks utiles.
    weighted_success = 0.0
    total_weight = 0.0

    weighted_delays = []
    weighted_errors = []
    weighted_low_losses = []

    delay_weights = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
    }

    # --------------------------------------------------------------
    # Analyse des expériences similaires
    # --------------------------------------------------------------

    for item in similar:

        mem = item["experience"]
        sim = float(item["similarity"])

        outcome = mem.get("outcome")

        if outcome == "gain":
            success_value = 1.0

        elif outcome in (
            "loss",
            "perte"
        ):
            success_value = 0.0

        else:
            continue

        weighted_success += (
            success_value * sim
        )

        total_weight += sim

        # ----------------------------------------------------------
        # Délai réel
        # ----------------------------------------------------------

        actual_delay = mem.get(
            "actual_delay"
        )

        if actual_delay is None:
            actual_delay = mem.get("actual_delay_sec", mem.get("actual_delay"))

        try:
            actual_delay = float(
                actual_delay
            )
        except (
            TypeError,
            ValueError
        ):
            actual_delay = None

        if (
            actual_delay is not None
            and math.isfinite(actual_delay)
            and 0 < actual_delay <= 1800
        ):
            weighted_delays.append(
                (
                    actual_delay,
                    sim
                )
            )

            bucket = _delay_bucket(
                actual_delay
            )

            if bucket in delay_weights:
                delay_weights[bucket] += sim

        # ----------------------------------------------------------
        # Erreur de cible
        # ----------------------------------------------------------

        target_error = mem.get(
            "target_error"
        )

        if target_error is not None:

            try:
                weighted_errors.append(
                    (
                        float(target_error),
                        sim
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                pass

        # ----------------------------------------------------------
        # Pertes de parcours
        # ----------------------------------------------------------

        low_losses = mem.get(
            "low_cote_losses"
        )

        if low_losses is not None:

            try:
                weighted_low_losses.append(
                    (
                        max(
                            0.0,
                            float(low_losses)
                        ),
                        sim
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                pass

    if total_weight <= 0:
        # Pas encore d'issue directionnelle exploitable : les données de
        # similarité restent insuffisantes pour modifier la prédiction.
        return empty

    # --------------------------------------------------------------
    # Score historique
    # --------------------------------------------------------------

    memory_score = (
        weighted_success
        / total_weight
    )

    # --------------------------------------------------------------
    # Délai moyen
    # --------------------------------------------------------------

    avg_delay = _weighted_average(
        weighted_delays
    )

    avg_delay_min = (
        avg_delay / 60.0
        if avg_delay is not None
        else None
    )

    # --------------------------------------------------------------
    # Médiane pondérée approximative
    # --------------------------------------------------------------

    median_delay = None

    if weighted_delays:

        ordered = sorted(
            weighted_delays,
            key=lambda x: x[0]
        )

        half = sum(
            weight
            for _, weight in ordered
        ) / 2.0

        cumulative = 0.0

        for value, weight in ordered:

            cumulative += weight

            if cumulative >= half:
                median_delay = value
                break

    # --------------------------------------------------------------
    # Distribution temporelle
    # --------------------------------------------------------------

    total_delay_weight = sum(
        delay_weights.values()
    )

    if total_delay_weight > 0:

        delay_distribution = {
            bucket: round(
                weight / total_delay_weight,
                6
            )
            for bucket, weight
            in delay_weights.items()
        }

        dominant_delay = max(
            delay_distribution,
            key=delay_distribution.get
        )

        dominant_probability = (
            delay_distribution[
                dominant_delay
            ]
        )

    else:

        delay_distribution = {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
        }

        dominant_delay = None
        dominant_probability = 0.0

    # --------------------------------------------------------------
    # Confiance temporelle
    # --------------------------------------------------------------

    if total_delay_weight > 0:

        timing_sample_confidence = clamp(
            total_delay_weight
            / MEMORY_HIGH_CONFIDENCE_SAMPLES,
            0.0,
            1.0
        )

        timing_concentration = clamp(
            dominant_probability,
            0.0,
            1.0
        )

        timing_confidence = clamp(
            0.55 * timing_sample_confidence
            + 0.45 * timing_concentration,
            0.0,
            1.0
        )

    else:
        timing_confidence = 0.0

    # --------------------------------------------------------------
    # Erreur moyenne de cible
    # --------------------------------------------------------------

    avg_target_error = _weighted_average(
        weighted_errors
    )

    # --------------------------------------------------------------
    # Pertes de parcours
    # --------------------------------------------------------------

    avg_low_cote_losses = _weighted_average(
        weighted_low_losses
    )

    # --------------------------------------------------------------
    # Confiance générale
    # --------------------------------------------------------------

    sample_conf = clamp(
        len(similar)
        / MEMORY_HIGH_CONFIDENCE_SAMPLES,
        0.0,
        1.0
    )

    sim_conf = clamp(
        total_weight
        / len(similar),
        0.0,
        1.0
    )

    confidence = clamp(
        0.60 * sample_conf
        + 0.40 * sim_conf,
        0.0,
        1.0
    )

    # Les premiers feedbacks sont utiles, mais leur poids augmente
    # progressivement au lieu d'attendre un seuil brutal.
    sample_strength = clamp(len(similar) / max(float(MEMORY_HIGH_CONFIDENCE_SAMPLES), 1.0), 0.0, 1.0)
    confidence *= sample_strength
    influence = MEMORY_MAX_INFLUENCE * confidence

    # Risque lié aux parcours de pertes <1.99x, estimé sur les expériences
    # similaires. Il est un signal de prudence, pas une certitude prédictive.
    low_loss_risk = (
        clamp((avg_low_cote_losses or 0.0) / 10.0, 0.0, 1.0)
        if avg_low_cote_losses is not None else None
    )

    return {
        "available": True,

        "samples": len(similar),

        "success_rate": round(
            memory_score,
            6
        ),

        "avg_delay": (
            round(avg_delay, 3)
            if avg_delay is not None
            else None
        ),

        "avg_delay_min": (
            round(avg_delay_min, 4)
            if avg_delay_min is not None
            else None
        ),

        "median_delay": (
            round(median_delay, 3)
            if median_delay is not None
            else None
        ),

        "dominant_delay": dominant_delay,

        "delay_distribution": delay_distribution,

        "timing_confidence": round(
            timing_confidence,
            6
        ),

        # Only measured elapsed times count here. Gain/Perte or +/-1 minute
        # qualitative feedback alone must not be mistaken for a measured time.
        "timing_samples": len(weighted_delays),
        "timing_weight": round(total_delay_weight, 6),

        "avg_target_error": (
            round(
                avg_target_error,
                3
            )
            if avg_target_error is not None
            else None
        ),

        "avg_low_cote_losses": (
            round(
                avg_low_cote_losses,
                3
            )
            if avg_low_cote_losses is not None
            else None
        ),

        "low_loss_risk": (
            round(low_loss_risk, 4)
            if low_loss_risk is not None
            else None
        ),

        "confidence": round(
            confidence,
            6
        ),

        "memory_score": round(
            memory_score,
            6
        ),

        "influence": round(
            influence,
            6
        ),
    }


def apply_memory_learning(
    ai_score: float,
    memory_result: dict
) -> float:

    if (
        not memory_result
        or not memory_result.get("available")
    ):
        return ai_score

    mem_score = clamp(
        float(
            memory_result.get(
                "memory_score",
                MEMORY_SUCCESS_NEUTRAL
            )
        ),
        0.0,
        1.0
    )

    influence = clamp(
        float(
            memory_result.get(
                "influence",
                0.0
            )
        ),
        0.0,
        MEMORY_MAX_INFLUENCE
    )

    adjustment = (
        mem_score
        - MEMORY_SUCCESS_NEUTRAL
    ) * influence

    return clamp(
        ai_score + adjustment,
        0.0,
        1.0
    )


# ══════════════════════════════════════════════════════════════════════
# MODULE 4 — FEEDBACK → EXPERIENCE MEMORY UPDATE
# ══════════════════════════════════════════════════════════════════════

def compute_real_delay(
    pred_dt: datetime.datetime,
    real_hour: int,
    real_minute: int
) -> float:

    real_dt = pred_dt.replace(
        hour=real_hour,
        minute=real_minute,
        second=0,
        microsecond=0
    )

    if real_dt < pred_dt:
        real_dt += datetime.timedelta(
            days=1
        )

    return (
        real_dt - pred_dt
    ).total_seconds()


def build_experience_record(
    prediction: dict,
    result_status: str,
    real_time: Optional[str] = None,
    real_max_cote: Optional[float] = None,
    delay_seconds: Optional[float] = None,
    low_cote_losses: int = 0,
) -> Optional[dict]:

    if not prediction:
        return None

    result = prediction.get(
        "result",
        {}
    )

    cotes = prediction.get(
        "cotes",
        []
    )

    last = (
        cotes[-ANALYSE_LAST:]
        if cotes
        else []
    )

    m, s = mean_std(last)

    low = (
        consecutive_low(last)
        if last
        else 0
    )

    markov_val = weighted_markov_next(
        last
    )

    markov_norm = (
        clamp(
            markov_val
            / (m + 1e-6),
            0.0,
            1.0
        )
        if m > 0
        else 0.0
    )

    pat = detect_weighted_patterns(
        last,
        pattern_memory,
        SEQ_WEIGHTS
    )

    peak = compute_peak_interval(
        last
    )

    # --------------------------------------------------------------
    # Délai
    # --------------------------------------------------------------

    actual_delay = None

    if delay_seconds is not None:

        try:
            actual_delay = float(
                delay_seconds
            )

            if not math.isfinite(actual_delay) or actual_delay < 0 or actual_delay > 1800:
                actual_delay = None

        except (
            TypeError,
            ValueError
        ):
            actual_delay = None

    predicted_delay = (
        result.get(
            "delay_sec",
            prediction.get(
                "delay_sec",
                0
            )
        )
    )

    try:
        predicted_delay = float(
            predicted_delay or 0
        )
    except (
        TypeError,
        ValueError
    ):
        predicted_delay = 0.0

    try:
        prediction_hour = datetime.datetime.fromisoformat(
            str(
                prediction.get(
                    "pred_time",
                    ""
                )
            )
        ).hour
    except Exception:
        prediction_hour = datetime.datetime.now().hour

    timing_error = None

    if actual_delay is not None:
        timing_error = (
            actual_delay
            - predicted_delay
        )

    return {
        "experience_id": hashlib.sha256(
            f"{time.time()}_{len(experience_memory)}"
            .encode()
        ).hexdigest()[:16],

        "created_at":
            datetime.datetime.now().isoformat(),

        "mode":
            prediction.get("mode"),

        "pred_time":
            prediction.get("pred_time"),

        "hour":
            prediction_hour,

        "prediction":
            result,

        "cotes":
            list(cotes),

        "last_cotes":
            list(last),

        "cote_count":
            len(cotes),

        "mean":
            float(m),

        "std":
            float(s),

        "volatility":
            float(
                volatility(last)
            ) if last else 0.0,

        "stability":
            float(
                stability_score(last)
            ) if last else 0.0,

        "low_series":
            int(low),

        "entropy":
            float(
                entropy_score(last)
            ) if last else 0.0,

        "bayes_probability":
            float(
                bayes_probability(last)
            ) if last else 0.0,

        "markov_prediction":
            float(markov_val),

        "markov_normalized":
            float(markov_norm),

        "pattern_net":
            float(
                pat.get("net", 0.0)
            ),

        "pattern_gain":
            float(
                pat.get("raw_gain", 0.0)
            ),

        "pattern_loss":
            float(
                pat.get("raw_loss", 0.0)
            ),

        "peak_interval":
            float(peak)
            if peak is not None
            else None,

        "ai_score":
            float(
                prediction.get(
                    "ai_score",
                    0.0
                )
            ),

        "status":
            result_status,

        "outcome":
            result_status,

        "real_time":
            real_time,

        "real_max_cote":
            real_max_cote,

        "delay_seconds": actual_delay,
        "actual_delay_sec": actual_delay,
        "delay_sec": predicted_delay,
        "predicted_delay_sec": predicted_delay,
        "predicted_delay": predicted_delay,
        "actual_delay": actual_delay,

        "timing_error":
            timing_error,

        "delay_error_seconds":
            timing_error,

        "low_cote_losses":
            max(
                0,
                int(
                    low_cote_losses or 0
                )
            ),

        "label":
            1
            if result_status == "gain"
            else 0,

        "reliability":
            EXP_RELIABILITY_DEFAULT,

        "verified_count":
            0,

        # ----------------------------------------------------------
        # Trace temporelle
        # ----------------------------------------------------------

        "timing_context_key":
            prediction.get(
                "timing_context_key"
            ),

        "delay_scores":
            dict(
                prediction.get(
                    "delay_scores",
                    {}
                ) or {}
            ),

        "timing_features":
            dict(
                prediction.get(
                    "timing_features",
                    {}
                ) or {}
            ),

        "timing_evidence":
            dict(
                prediction.get(
                    "timing_evidence",
                    {}
                ) or {}
            ),
    }


def update_experience_memory(
    prediction: dict,
    status: str,
    real_time: Optional[str] = None,
    real_max_cote: Optional[float] = None,
    delay_seconds: Optional[float] = None,
    low_cote_losses: int = 0,
) -> Optional[dict]:

    if not prediction:
        return None

    existing_id = prediction.get(
        "experience_id"
    )

    # ══════════════════════════════════════════════════════════════
    # CAS 1 — expérience existante
    # ══════════════════════════════════════════════════════════════

    if existing_id:

        for existing in experience_memory:

            if not isinstance(
                existing,
                dict
            ):
                continue

            if (
                existing.get(
                    "experience_id"
                )
                != existing_id
            ):
                continue

            result = prediction.get(
                "result",
                {}
            )

            try:

                pred_dt = (
                    datetime.datetime
                    .fromisoformat(
                        prediction.get(
                            "pred_time",
                            ""
                        )
                    )
                )

            except Exception:

                pred_dt = (
                    datetime.datetime.now()
                )

            actual_dt = None

            if real_time:

                try:
                    actual_dt = (
                        parse_real_datetime(
                            real_time,
                            pred_dt
                        )
                    )
                except Exception:
                    actual_dt = None

            target = float(
                result.get(
                    "target",
                    0.0
                )
                or 0.0
            )

            actual_max = float(
                real_max_cote
                or 0.0
            )

            # ------------------------------------------------------
            # Calcul sécurisé du délai réel
            # ------------------------------------------------------

            actual_delay = None

            if delay_seconds is not None:

                try:

                    actual_delay = float(
                        delay_seconds
                    )

                    if not math.isfinite(actual_delay) or actual_delay < 0 or actual_delay > 1800:
                        actual_delay = None

                except (
                    TypeError,
                    ValueError
                ):
                    actual_delay = None

            predicted_delay = float(
                existing.get(
                    "predicted_delay",
                    result.get(
                        "delay_sec",
                        0.0
                    )
                    or 0.0
                )
                or 0.0
            )

            timing_error = None

            if actual_delay is not None:

                timing_error = (
                    actual_delay
                    - predicted_delay
                )

            # ------------------------------------------------------
            # Classe temporelle réelle
            # ------------------------------------------------------

            timing_bucket = _delay_bucket(
                actual_delay
            )

            # ------------------------------------------------------
            # Mise à jour
            # ------------------------------------------------------

            existing.update({

                "status":
                    status,

                "outcome":
                    status,

                "real_time":
                    real_time,

                "actual_time":
                    (
                        actual_dt.isoformat()
                        if actual_dt
                        else ""
                    ),

                "real_max_cote":
                    actual_max,

                "actual_max_cote":
                    actual_max,

                "actual_delay": actual_delay,
                "actual_delay_sec": actual_delay,
                "delay_seconds": actual_delay,
                "delay_sec": existing.get("delay_sec", predicted_delay),
                "predicted_delay_sec": predicted_delay,
                "predicted_delay":
                    predicted_delay,

                "timing_error":
                    timing_error,

                "delay_error_seconds":
                    timing_error,

                "timing_bucket":
                    timing_bucket,

                "actual_delay_min":
                    (
                        actual_delay / 60.0
                        if actual_delay is not None
                        else None
                    ),

                "timing_correct":
                    (
                        abs(timing_error) <= 30.0
                        if timing_error is not None
                        else None
                    ),

                "low_cote_losses":
                    max(
                        0,
                        int(
                            low_cote_losses
                            or 0
                        )
                    ),

                "target_error":
                    (
                        actual_max - target
                        if target
                        else 0.0
                    ),

                "updated_at":
                    datetime.datetime.now().isoformat(),
            })

            # ------------------------------------------------------
            # Mise à jour de la fiabilité
            # ------------------------------------------------------

            old_count = int(
                existing.get(
                    "verified_count",
                    0
                )
                or 0
            )

            existing["verified_count"] = (
                old_count + 1
            )

            # Une expérience vérifiée gagne progressivement
            # davantage de poids dans la mémoire.
            old_reliability = float(
                existing.get(
                    "reliability",
                    EXP_RELIABILITY_DEFAULT
                )
            )

            if status == "gain":
                feedback_value = 1.0
            elif status in (
                "loss",
                "perte"
            ):
                feedback_value = 0.0
            else:
                feedback_value = 0.5

            existing["reliability"] = round(
                clamp(
                    0.90 * old_reliability
                    + 0.10 * feedback_value,
                    0.05,
                    0.99
                ),
                6
            )

            logging.info(
                "[MEMORY] Finalized existing "
                f"experience | id={existing_id[:8]} "
                f"status={status} "
                f"actual_delay={actual_delay} "
                f"timing_error={timing_error} "
                f"bucket={timing_bucket}"
            )

            return existing

    # ══════════════════════════════════════════════════════════════
    # CAS 2 — ancienne prédiction sans expérience attachée
    # ══════════════════════════════════════════════════════════════

    exp = build_experience_record(
        prediction=prediction,
        result_status=status,
        real_time=real_time,
        real_max_cote=real_max_cote,
        delay_seconds=delay_seconds,
        low_cote_losses=low_cote_losses,
    )

    if exp is None:

        logging.warning(
            "[MEMORY] Experience creation failed"
        )

        return None

    stored = store_experience(
        exp
    )

    if not stored:
        return None

    # --------------------------------------------------------------
    # Apprentissage immédiat de cette expérience
    # --------------------------------------------------------------

    mem_pred = compute_memory_prediction(
        exp
    )

    exp["memory_probability"] = (
        mem_pred["probability"]
    )

    exp["memory_confidence"] = (
        mem_pred["confidence"]
    )

    exp["similar_samples"] = (
        mem_pred["samples"]
    )

    logging.info(
        "[MEMORY] Stored | "
        f"status={status} | "
        f"mode={exp.get('mode')} | "
        f"mem_prob={mem_pred['probability']:.3f} | "
        f"samples={mem_pred['samples']} | "
        f"actual_delay={exp.get('actual_delay')}"
    )

    return exp


# ══════════════════════════════════════════════════════════════════════
# MODULE 5 — MEMORY-AWARE AI SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_memory_weight(
    memory_probability,
    memory_confidence,
    samples
) -> float:

    if samples < MEMORY_MIN_SAMPLES_M5:
        return 0.0

    if (
        memory_confidence
        < MEMORY_CONFIDENCE_THRESHOLD
    ):
        return 0.0

    sample_factor = min(
        1.0,
        samples / 20.0
    )

    conf_factor = clamp(
        memory_confidence,
        0.0,
        1.0
    )

    return clamp(
        MEMORY_MAX_WEIGHT
        * sample_factor
        * conf_factor,
        0.0,
        MEMORY_MAX_WEIGHT
    )


def build_experience_query_legacy(
    cotes,
    mode,
    ai_score
) -> Optional[dict]:

    if not cotes:
        return None

    last = cotes[-ANALYSE_LAST:]

    m, s = mean_std(last)

    low = consecutive_low(
        last
    )

    markov_val = weighted_markov_next(
        last
    )

    markov_norm = (
        clamp(
            markov_val
            / (m + 1e-6),
            0.0,
            1.0
        )
        if m > 0
        else 0.0
    )

    pat = detect_weighted_patterns(
        last,
        pattern_memory,
        SEQ_WEIGHTS
    )

    peak = compute_peak_interval(
        last
    )

    return {

        "mode":
            mode,

        "cotes":
            list(cotes),

        "last_cotes":
            list(last),

        "mean":
            float(m),

        "std":
            float(s),

        "volatility":
            float(
                volatility(last)
            ),

        "stability":
            float(
                stability_score(last)
            ),

        "low_series":
            int(low),

        "entropy":
            float(
                entropy_score(last)
            ),

        "bayes_probability":
            float(
                bayes_probability(last)
            ),

        "markov_prediction":
            float(markov_val),

        "markov_normalized":
            float(markov_norm),

        "pattern_net":
            float(
                pat.get(
                    "net",
                    0.0
                )
            ),

        "pattern_gain":
            float(
                pat.get(
                    "raw_gain",
                    0.0
                )
            ),

        "pattern_loss":
            float(
                pat.get(
                    "raw_loss",
                    0.0
                )
            ),

        "peak_interval":
            (
                float(peak)
                if peak is not None
                else None
            ),

        "ai_score":
            float(ai_score),
    }


def compute_memory_prediction(
    current_exp
) -> dict:

    if not current_exp:
        return {
            "probability": 0.5,
            "confidence": 0.0,
            "samples": 0,
        }

    # --------------------------------------------------------------
    # Utilise le même moteur de similarité que le nouveau système.
    # --------------------------------------------------------------

    similar = find_similar_experiences_v2(
        current_exp,
        limit=50
    )

    if not similar:
        return {
            "probability": 0.5,
            "confidence": 0.0,
            "samples": 0,
        }

    weighted_gain = 0.0
    weighted_total = 0.0

    valid_samples = 0

    for item in similar:

        exp = item["experience"]

        sim = float(
            item["similarity"]
        )

        if sim < MEMORY_MIN_SIMILARITY:
            continue

        outcome = exp.get(
            "outcome",
            exp.get("status")
        )

        if outcome == "gain":

            weighted_gain += sim
            weighted_total += sim
            valid_samples += 1

        elif outcome in (
            "loss",
            "perte"
        ):

            weighted_total += sim
            valid_samples += 1

    if weighted_total <= 0:

        return {
            "probability": 0.5,
            "confidence": 0.0,
            "samples": 0,
        }

    probability = (
        weighted_gain
        / weighted_total
    )

    confidence = clamp(
        (
            0.60
            * min(
                1.0,
                valid_samples / 20.0
            )
        )
        +
        (
            0.40
            * min(
                1.0,
                weighted_total
                / 10.0
            )
        ),
        0.0,
        1.0
    )

    return {

        "probability":
            round(
                probability,
                4
            ),

        "confidence":
            round(
                confidence,
                4
            ),

        "samples":
            valid_samples,
    }


def memory_based_prediction(
    cotes,
    mode,
    ai_score
) -> dict:

    current = build_current_experience(
        cotes,
        mode,
        ai_score
    )

    if current is None:

        return {
            "probability": 0.5,
            "confidence": 0.0,
            "weight": 0.0,
            "samples": 0,
            "active": False,
            "timing": {
                "available": False
            },
        }

    # --------------------------------------------------------------
    # Résultat historique
    # --------------------------------------------------------------

    mem = compute_memory_prediction(
        current
    )

    prob = mem.get(
        "probability",
        0.5
    )

    conf = mem.get(
        "confidence",
        0.0
    )

    samp = mem.get(
        "samples",
        0
    )

    weight = compute_memory_weight(
        prob,
        conf,
        samp
    )

    # --------------------------------------------------------------
    # Apprentissage temporel séparé
    # --------------------------------------------------------------

    timing_memory = compute_memory_learning(
        current
    )

    return {

        "probability":
            clamp(
                prob,
                0.0,
                1.0
            ),

        "confidence":
            clamp(
                conf,
                0.0,
                1.0
            ),

        "weight":
            weight,

        "samples":
            samp,

        "active":
            weight > 0.0,

        # ----------------------------------------------------------
        # NOUVEAU :
        # Le résultat temporel n'est PAS mélangé directement
        # dans le score directionnel.
        # ----------------------------------------------------------

        "timing":
            timing_memory,
    }


# ══════════════════════════════════════════════════════════════════════
# MODULE — DYNAMIC CONTEXT & CONTEXT MEMORY
# ══════════════════════════════════════════════════════════════════════

def detect_market_context(
    cotes: list
) -> str:

    if not cotes or len(cotes) < 5:
        return "CALM"

    recent = cotes[-15:]

    ecart_type = float(
        np.std(recent)
    )

    small = sum(
        1
        for x in recent
        if x < 1.50
    )

    big = sum(
        1
        for x in recent
        if x >= 5.0
    )

    last_spike_index = None

    for i in range(
        len(recent) - 1,
        -1,
        -1
    ):

        if recent[i] >= 5.0:
            last_spike_index = i
            break

    if last_spike_index is not None:

        tours_after_spike = (
            len(recent)
            - 1
            - last_spike_index
        )

        if tours_after_spike <= 5:
            return "POST_SPIKE"

    if (
        big >= 2
        or max(recent) >= 10.0
    ):
        return "EXPLOSIVE"

    taux_petites = (
        small
        / len(recent)
    )

    if (
        taux_petites >= 0.70
        and ecart_type < 1.50
    ):
        return "COMPRESSED"

    if ecart_type >= 2.50:
        return "VOLATILE"

    return "CALM"


def create_context_memory() -> dict:

    return {
        ctx: {
            "gains": {},
            "losses": {},
            "last_seen": {},
            "confidence": {},
        }
        for ctx in CONTEXTS
    }


def build_pattern_key(
    cotes: list
) -> str:

    if not cotes or len(cotes) < 5:
        return "UNKNOWN"

    recent = cotes[-10:]

    pattern = []

    for cote in recent:

        if cote < 1.30:
            pattern.append("L")

        elif cote < 2.00:
            pattern.append("M")

        elif cote < 5.00:
            pattern.append("H")

        else:
            pattern.append("X")

    return "_".join(pattern)


def update_context_memory(
    context: str,
    pattern_key: str,
    success: bool,
    confidence: float
) -> None:

    memory = AI_STATE.get(
        "pattern_memory"
    )

    if memory is None:

        AI_STATE["pattern_memory"] = (
            create_context_memory()
        )

        memory = AI_STATE[
            "pattern_memory"
        ]

    if context not in memory:

        memory[context] = {
            "gains": {},
            "losses": {},
            "last_seen": {},
            "confidence": {},
        }

    if success:

        memory[context]["gains"][
            pattern_key
        ] = (
            memory[context]["gains"].get(
                pattern_key,
                0
            )
            + 1
        )

        AI_STATE[
            "total_gains"
        ] = (
            AI_STATE.get(
                "total_gains",
                0
            )
            + 1
        )

    else:

        memory[context]["losses"][
            pattern_key
        ] = (
            memory[context]["losses"].get(
                pattern_key,
                0
            )
            + 1
        )

        AI_STATE[
            "total_losses"
        ] = (
            AI_STATE.get(
                "total_losses",
                0
            )
            + 1
        )

    memory[context][
        "confidence"
    ][pattern_key] = clamp(
        float(confidence),
        0.0,
        1.0
    )

    memory[context][
        "last_seen"
    ][pattern_key] = time.time()


def get_pattern_reliability(
    context: str,
    pattern_key: str
) -> float:

    memory = AI_STATE.get(
        "pattern_memory",
        {}
    ).get(
        context,
        {}
    )

    gains = memory.get(
        "gains",
        {}
    ).get(
        pattern_key,
        0
    )

    losses = memory.get(
        "losses",
        {}
    ).get(
        pattern_key,
        0
    )

    total = gains + losses

    if total == 0:
        return 0.50

    return clamp(
        gains / total,
        0.0,
        1.0
    )


# ══════════════════════════════════════════════════════════════════════
# MODULE — DYNAMIC PROBABILITY & MULTI-SOURCE SCORE
# ══════════════════════════════════════════════════════════════════════

def historical_probability() -> float:

    gains = AI_STATE.get(
        "total_gains",
        0
    )

    losses = AI_STATE.get(
        "total_losses",
        0
    )

    total = gains + losses

    if total == 0:
        return 0.50

    return gains / total


def context_probability(
    context: str
) -> float:

    memory = AI_STATE.get(
        "pattern_memory",
        {}
    ).get(
        context,
        {}
    )

    gains = sum(
        memory.get(
            "gains",
            {}
        ).values()
    )

    losses = sum(
        memory.get(
            "losses",
            {}
        ).values()
    )

    total = gains + losses

    if total == 0:
        return 0.50

    return gains / total


def smoothed_probability(
    gains: int,
    losses: int,
    prior: float = 0.50,
    strength: int = 10
) -> float:

    total = gains + losses

    if total == 0:
        return prior

    raw_prob = (
        gains / total
    )

    confidence_factor = (
        total
        / (total + strength)
    )

    return (
        raw_prob
        * confidence_factor
        +
        prior
        * (1 - confidence_factor)
    )


def sequence_probability(
    cotes: list
) -> float:

    if not cotes:
        return 0.50

    recent = cotes[-10:]

    low_count = sum(
        1
        for x in recent
        if x < 1.50
    )

    ratio_low = (
        low_count
        / len(recent)
    )

    prob = 0.50

    if ratio_low >= 0.80:
        prob += 0.08

    elif ratio_low >= 0.60:
        prob += 0.04

    elif ratio_low <= 0.20:
        prob -= 0.04

    return clamp(
        prob,
        0.05,
        0.95
    )


def combine_probabilities(
    historical_prob: float,
    context_prob: float,
    pattern_prob: float,
    sequence_prob: float,
    neural_prob: float = None,
) -> float:

    probabilities = [
        historical_prob,
        context_prob,
        pattern_prob,
        sequence_prob,
    ]

    weights = [
        0.20,
        0.20,
        0.30,
        0.15,
    ]

    if neural_prob is not None:

        probabilities.append(
            neural_prob
        )

        weights.append(
            0.15
        )

    total_weight = sum(
        weights
    )

    final_probability = (
        sum(
            p * w
            for p, w in zip(
                probabilities,
                weights
            )
        )
        / total_weight
    )

    return clamp(
        final_probability,
        0.01,
        0.99
    )


def calculate_entropy(
    cotes: list
) -> float:

    if not cotes:
        return 1.0

    recent = cotes[-15:]

    total = len(recent)

    low = sum(
        1
        for x in recent
        if x < 1.50
    )

    medium = sum(
        1
        for x in recent
        if 1.50 <= x < 5.0
    )

    high = sum(
        1
        for x in recent
        if x >= 5.0
    )

    entropy = 0.0

    for p in (
        low / total,
        medium / total,
        high / total,
    ):

        if p > 0:
            entropy -= (
                p * math.log2(p)
            )

    max_entropy = math.log2(3)

    return entropy / max_entropy


def calculate_final_score(
    probability: float,
    entropy: float,
    data_quality: float = 1.0
) -> float:

    uncertainty_penalty = (
        entropy * 0.15
    )

    final_score = (
        probability
        - uncertainty_penalty
    ) * data_quality

    return clamp(
        final_score,
        0.01,
        0.99
    )


def calculate_dynamic_ai_score(
    cotes: list,
    neural_probability: float = None
) -> dict:

    if (
        not cotes
        or len(cotes) < 5
    ):

        return {
            "score": 0.50,
            "probability": 0.50,
            "context": "UNKNOWN",
            "pattern": "UNKNOWN",
            "entropy": 1.0,
        }

    context = detect_market_context(
        cotes
    )

    pattern_key = build_pattern_key(
        cotes
    )

    historical_prob = (
        historical_probability()
    )

    context_prob = (
        context_probability(
            context
        )
    )

    pattern_prob = (
        get_pattern_reliability(
            context,
            pattern_key
        )
    )

    sequence_prob = (
        sequence_probability(
            cotes
        )
    )

    combined_prob = combine_probabilities(
        historical_prob,
        context_prob,
        pattern_prob,
        sequence_prob,
        neural_probability,
    )

    entropy = calculate_entropy(
        cotes
    )

    final_score = calculate_final_score(
        combined_prob,
        entropy
    )

    return {

        "score":
            final_score,

        "probability":
            combined_prob,

        "context":
            context,

        "pattern":
            pattern_key,

        "entropy":
            entropy,

        "historical_probability":
            historical_prob,

        "context_probability":
            context_prob,

        "pattern_probability":
            pattern_prob,

        "sequence_probability":
            sequence_prob,

        "neural_probability":
            neural_probability,
    }


# ══════════════════════════════════════════════════════════════════════
# MODULE — ADAPTIVE MODEL WEIGHTING
# ══════════════════════════════════════════════════════════════════════

def create_model_performance() -> dict:

    return {
        ctx: {
            model: {
                "wins": 0,
                "losses": 0,
                "total": 0,
                "reliability": 0.50,
            }
            for model in MODEL_NAMES
        }
        for ctx in CONTEXTS
    }


def update_model_performance(
    context: str,
    model_name: str,
    success: bool
) -> None:

    performance = AI_STATE.get(
        "model_performance"
    )

    if performance is None:

        AI_STATE[
            "model_performance"
        ] = create_model_performance()

        performance = AI_STATE[
            "model_performance"
        ]

    if context not in performance:
        return

    if model_name not in performance[
        context
    ]:
        return

    stats = performance[
        context
    ][model_name]

    stats["total"] += 1

    if success:
        stats["wins"] += 1
    else:
        stats["losses"] += 1

    stats["reliability"] = (
        smoothed_probability(
            stats["wins"],
            stats["losses"],
            prior=0.50,
            strength=10,
        )
    )


def get_adaptive_weights(
    context: str
) -> dict:

    performance = AI_STATE.get(
        "model_performance",
        {}
    ).get(
        context,
        {}
    )

    raw_weights = {}

    for model_name in MODEL_NAMES:

        stats = performance.get(
            model_name,
            {}
        )

        reliability = stats.get(
            "reliability",
            0.50
        )

        total = stats.get(
            "total",
            0
        )

        confidence_factor = min(
            1.0,
            total / 50.0
        )

        adaptive_reliability = (
            0.50
            * (1 - confidence_factor)
            +
            reliability
            * confidence_factor
        )

        raw_weights[
            model_name
        ] = adaptive_reliability

    total_weight = sum(
        raw_weights.values()
    )

    if total_weight <= 0:

        return {
            m: 1 / len(MODEL_NAMES)
            for m in MODEL_NAMES
        }

    return {
        m: w / total_weight
        for m, w in raw_weights.items()
    }


def adaptive_probability_fusion(
    context: str,
    probabilities: dict
) -> float:

    weights = get_adaptive_weights(
        context
    )

    total = 0.0
    weight_sum = 0.0

    for model_name in MODEL_NAMES:

        prob = probabilities.get(
            model_name,
            0.50
        )

        weight = weights.get(
            model_name,
            0.0
        )

        total += (
            prob * weight
        )

        weight_sum += weight

    if weight_sum == 0:
        return 0.50

    return clamp(
        total / weight_sum,
        0.01,
        0.99
    )


# ══════════════════════════════════════════════════════════════════════
# MODULE — FINAL AI SCORE
# ══════════════════════════════════════════════════════════════════════

def get_model_rich_output(
    last,
    mode,
    context,
    pattern_data,
    seq_weights,
    cotes
) -> dict:
    """
    Construit les sorties normalisées consommées par le Meta Decision Engine.

    Cette étape doit rester strictement pré-prédiction : elle ne lit que les
    cotes disponibles, les modèles entraînés et les mémoires déjà validées.
    """

    recent = list(
        last
        or cotes
        or []
    )[-ANALYSE_LAST:]

    if not recent:
        recent = [1.0]

    mean_value, std_value = mean_std(recent)
    low_value = consecutive_low(recent)
    volatility_value = volatility(recent)
    stability_value = stability_score(recent)
    entropy_value = entropy_score(recent)
    markov_value = weighted_markov_next(recent)
    markov_probability = clamp(
        markov_value / (mean_value + 1e-6)
        if mean_value > 0
        else 0.0,
        0.0,
        1.0
    )

    bayes_probability_value = clamp(
        bayes_probability(
            recent,
            threshold=2.0
        ),
        0.0,
        1.0
    )

    pattern_result = detect_weighted_patterns(
        recent,
        pattern_data,
        seq_weights
    )
    pattern_signal = clamp(
        max(
            0.0,
            _safe_float(
                pattern_result.get(
                    "net",
                    0.0
                )
            )
        ),
        0.0,
        1.0
    )

    pattern_key = build_pattern_key(
        cotes or recent
    )
    pattern_reliability = clamp(
        get_pattern_reliability(
            context,
            pattern_key
        ),
        0.0,
        1.0
    )
    pattern_probability_value = (
        pattern_signal * 0.60
        + pattern_reliability * 0.40
    )

    neural_features = build_prediction_features(
        consecutive_low_value=low_value,
        volatility_value=volatility_value,
        std_value=std_value,
        mean_value=mean_value,
        predicted_score=0.0,
        predicted_target=0.0,
        predicted_hour=datetime.datetime.now().hour,
        predicted_minute=datetime.datetime.now().minute,
        peak_interval=(
            compute_peak_interval(recent)
            or 0.0
        ),
        sequence_length=len(recent),
        pattern_gain=_safe_float(
            pattern_result.get(
                "raw_gain",
                0.0
            )
        ),
        pattern_loss=_safe_float(
            pattern_result.get(
                "raw_loss",
                0.0
            )
        ),
    )
    neural_probability_value = clamp(
        predict_with_model(neural_features),
        0.0,
        1.0
    )

    memory_probability_value = MEMORY_SUCCESS_NEUTRAL
    memory_confidence = 0.0

    try:
        current_context = build_current_context(
            cotes or recent,
            mode,
            0.50
        )
        memory_result = compute_memory_learning(
            current_context
        )
        memory_probability_value = clamp(
            _safe_float(
                memory_result.get(
                    "memory_score",
                    memory_result.get(
                        "success_rate",
                        MEMORY_SUCCESS_NEUTRAL
                    )
                ),
                MEMORY_SUCCESS_NEUTRAL
            ),
            0.0,
            1.0
        )
        memory_confidence = clamp(
            _safe_float(
                memory_result.get(
                    "confidence",
                    0.0
                )
            ),
            0.0,
            1.0
        )
    except Exception as memory_error:
        logging.debug(
            "[META_ENGINE] memory source unavailable: %s",
            memory_error
        )

    historical_probability_value = clamp(
        historical_probability(),
        0.0,
        1.0
    )
    context_probability_value = clamp(
        context_probability(context),
        0.0,
        1.0
    )
    neural_confidence_value = clamp(
        neural_confidence(),
        0.0,
        1.0
    )
    sample_count = len(recent)

    def output(probability, confidence, samples=None):
        confidence = clamp(
            _safe_float(
                confidence,
                0.50
            ),
            0.0,
            1.0
        )
        return {
            "probability": clamp(
                _safe_float(
                    probability,
                    0.50
                ),
                0.0,
                1.0
            ),
            "confidence": confidence,
            "samples": (
                sample_count
                if samples is None
                else max(
                    0,
                    int(
                        _safe_float(
                            samples,
                            0
                        )
                    )
                )
            ),
            "uncertainty": 1.0 - confidence,
        }

    return {
        "markov": output(
            markov_probability,
            0.50 + abs(
                markov_probability - 0.50
            ),
        ),
        "bayes": output(
            bayes_probability_value,
            0.50 + min(
                0.50,
                len(recent) / 20.0
            ),
        ),
        "pattern": output(
            pattern_probability_value,
            0.50 + min(
                0.50,
                abs(
                    pattern_signal - 0.50
                )
            ),
        ),
        "neural": output(
            neural_probability_value,
            neural_confidence_value,
        ),
        "memory": output(
            memory_probability_value,
            memory_confidence,
            samples=(
                memory_result.get(
                    "samples",
                    0
                )
                if "memory_result" in locals()
                else 0
            ),
        ),
        "historical": output(
            historical_probability_value,
            0.50,
        ),
    }


def compute_ai_score(
    cotes,
    pattern_data,
    seq_weights,
    mode="2X-10X"
) -> float:

    if not cotes:
        return 0.0

    last = cotes[
        -ANALYSE_LAST:
    ]

    m, s = mean_std(
        last
    )

    low = consecutive_low(
        last
    )

    stab = stability_score(
        last
    )

    ent = entropy_score(
        last
    )

    bayes = bayes_probability(
        last,
        threshold=2.0
    )

    markov = weighted_markov_next(
        last
    )

    markov_norm = (
        clamp(
            markov
            / (m + 1e-6),
            0.0,
            1.0
        )
        if m > 0
        else 0.0
    )

    pat = detect_weighted_patterns(
        last,
        pattern_data,
        seq_weights
    )

    pat_score = clamp(
        max(
            0.0,
            pat["net"]
        ),
        0.0,
        1.0
    )

    peak_interval = (
        compute_peak_interval(
            last
        )
        or 0.0
    )

    # --------------------------------------------------------------
    # FEATURES STRICTEMENT PRÉ-PREDICTION
    # --------------------------------------------------------------

    features = build_prediction_features(
        consecutive_low_value=low,
        volatility_value=volatility(last),
        std_value=s,
        mean_value=m,
        predicted_score=0.0,
        predicted_target=0.0,
        predicted_hour=datetime.datetime.now().hour,
        predicted_minute=datetime.datetime.now().minute,
        peak_interval=peak_interval,
        sequence_length=len(last),
        pattern_gain=pat["raw_gain"],
        pattern_loss=pat["raw_loss"],
    )

    # --------------------------------------------------------------
    # NEURAL
    # --------------------------------------------------------------

    nn_raw = predict_with_model(
        features
    )

    nn_score = (
        _calibrate(nn_raw)
        if _calibrator_trained
        else nn_raw
    )

    # --------------------------------------------------------------
    # HEURISTIQUE
    # --------------------------------------------------------------

    heuristic = compute_learned_score(
        low=clamp(
            low / 10.0,
            0.0,
            1.0
        ),
        stability=stab,
        entropy=1.0 - ent,
        bayes=bayes,
        markov=clamp(
            markov_norm,
            0.0,
            1.0
        ),
        pattern=pat_score,
    )

    # --------------------------------------------------------------
    # COMBINAISON NN / HEURISTIQUE
    # --------------------------------------------------------------

    if len(training_dataset) < 50:

        base = (
            0.70 * heuristic
            +
            0.30 * nn_score
        )

    else:

        base = (
            0.40 * heuristic
            +
            0.60 * nn_score
        )

    # --------------------------------------------------------------
    # SCORE DYNAMIQUE
    # --------------------------------------------------------------

    dyn = calculate_dynamic_ai_score(
        cotes,
        neural_probability=nn_score
    )

    context = dyn["context"]

    pattern_key = dyn[
        "pattern"
    ]

    pattern_reliability = (
        get_pattern_reliability(
            context,
            pattern_key
        )
    )

    # --------------------------------------------------------------
    # FUSION ADAPTATIVE
    # --------------------------------------------------------------

    prob_sources = {

        "historical":
            dyn[
                "historical_probability"
            ],

        "context":
            dyn[
                "context_probability"
            ],

        "pattern":
            dyn[
                "pattern_probability"
            ],

        "sequence":
            dyn[
                "sequence_probability"
            ],

        "neural":
            nn_score,
    }

    adaptive_prob = (
        adaptive_probability_fusion(
            context,
            prob_sources
        )
    )

    # --------------------------------------------------------------
    # META DECISION ENGINE
    # --------------------------------------------------------------

    try:

        _model_out = get_model_rich_output(
            last,
            mode,
            context,
            pattern_data,
            seq_weights,
            cotes
        )

        _meta = meta_decision_engine(
            _model_out,
            context
        )

        meta_prob = _meta[
            "probability"
        ]

        final = (
            base * 0.35
            +
            meta_prob * 0.50
            +
            dyn["score"] * 0.15
        )

        _last_prediction_meta[
            "model_outputs"
        ] = _model_out

        _last_prediction_meta[
            "meta_result"
        ] = _meta

        _last_prediction_meta[
            "context"
        ] = context

        logging.info(
            "[META_ENGINE] "
            f"ctx={context} "
            f"top={_meta['top_model']} "
            f"meta_prob={meta_prob:.3f} "
            f"vote_high={_meta['vote_high']:.2f} "
            f"final_pre_mem={final:.3f}"
        )

    except Exception as _meta_e:

        logging.warning(
            f"[META_ENGINE] fallback: {_meta_e}"
        )

        final = (
            base * 0.50
            +
            pattern_reliability * 0.20
            +
            adaptive_prob * 0.15
            +
            dyn["score"] * 0.15
        )

    # --------------------------------------------------------------
    # MÉMOIRE
    # --------------------------------------------------------------

    mem_result = memory_based_prediction(
        cotes,
        mode,
        final
    )

    mem_prob = mem_result[
        "probability"
    ]

    mem_weight = mem_result[
        "weight"
    ]

    # La mémoire historique modifie légèrement le score
    # directionnel uniquement lorsqu'elle possède suffisamment
    # d'expériences vérifiées.
    final = (
        final * (1.0 - mem_weight)
        +
        mem_prob * mem_weight
    )

    final = clamp(
        final,
        0.0,
        1.0
    )

    # --------------------------------------------------------------
    # CACHE DU CONTEXTE MÉMOIRE
    # --------------------------------------------------------------

    _last_prediction_meta[
        "memory"
    ] = mem_result

    _last_prediction_meta[
        "memory_timing"
    ] = mem_result.get(
        "timing",
        {}
    )

    _last_prediction_meta[
        "timing_learning_available"
    ] = bool(
        mem_result.get(
            "timing",
            {}
        ).get(
            "available",
            False
        )
    )

    logging.info(
        "[AI SCORE] "
        f"ctx={context} "
        f"heuristic={heuristic:.3f} "
        f"nn={nn_score:.3f} "
        f"base={base:.3f} "
        f"adaptive={adaptive_prob:.3f} "
        f"dyn={dyn['score']:.3f} "
        f"pat_rel={pattern_reliability:.3f} "
        f"mem_w={mem_weight:.3f} "
        f"final={final:.3f}"
    )

    timing = mem_result.get(
        "timing",
        {}
    )

    if timing.get("available"):

        logging.info(
            "[TIMING MEMORY] "
            f"samples={timing.get('samples')} "
            f"avg_delay={timing.get('avg_delay')} "
            f"dominant={timing.get('dominant_delay')} "
            f"timing_conf={timing.get('timing_confidence')}"
        )

    return final

# ══════════════════════════════════════════════════════════════════════
# MODULE 6 — EXPERIENCE MEMORY RETRIEVAL — CORRECTED
# ══════════════════════════════════════════════════════════════════════

def _historical_sequence_similarity(current_sequence, historical_sequence, length):
    """Compare aligned recent crash multipliers in log space (0..1)."""
    try:
        cur = [max(1.0, float(x)) for x in list(current_sequence or [])[-length:]]
        old = [max(1.0, float(x)) for x in list(historical_sequence or [])[-length:]]
        n = min(len(cur), len(old))
        if n < max(3, min(length, 5)):
            return None
        cur, old = cur[-n:], old[-n:]
        weights = [0.5 + (i + 1) / n for i in range(n)]
        diffs = [min(1.0, abs(math.log(a) - math.log(b)) / math.log(10.0)) for a, b in zip(cur, old)]
        distance = sum(d * w for d, w in zip(diffs, weights)) / sum(weights)
        return clamp(1.0 - distance, 0.0, 1.0)
    except Exception:
        return None


def _historical_timestamp(record):
    """Read only the timestamp stored when the historical prediction was made."""
    if not isinstance(record, dict):
        return None
    for key in ("created_at", "timestamp", "prediction_datetime", "predicted_at"):
        value = record.get(key)
        if not value:
            continue
        try:
            dt = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt
        except Exception:
            continue
    try:
        hour = int(record.get("hour"))
        minute = int(record.get("minute", 0))
        return datetime.datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    except Exception:
        return None


def historical_comparison_engine(current_context, now=None, limit=MAX_EXPERIENCE_MEMORY):
    """Build read-only historical cohorts; never lets pending records enter."""
    empty_cohort = {"samples": 0, "weighted_gain_rate": None, "avg_target_error": None,
                    "avg_timing_error": None, "avg_actual_delay_sec": None,
                    "effective_weight": 0.0, "mean_similarity": None}
    result = {
        "version": 1,
        "status": "insufficient_data",
        "sequence_lengths": [5, 10, 15],
        "windows_minutes": [5, 15, 30],
        "all_similar": dict(empty_cohort),
        "time_windows": {str(w): dict(empty_cohort) for w in (5, 15, 30)},
    }
    if not isinstance(current_context, dict) or not current_context:
        return result

    now = now or datetime.datetime.now()
    current_seq = current_context.get("sequence", current_context.get("cotes", [])) or []
    current_mode = current_context.get("mode")
    if not current_seq or not isinstance(experience_memory, list):
        return result

    scored = []
    for record in experience_memory[-max(1, int(limit)):]:
        if not isinstance(record, dict) or record.get("mode") != current_mode:
            continue
        outcome = str(record.get("outcome", record.get("status", ""))).lower()
        if outcome not in ("gain", "loss", "perte"):
            continue

        historical_seq = record.get("sequence", record.get("cotes", [])) or []
        seq_scores = []
        for length in (5, 10, 15):
            value = _historical_sequence_similarity(current_seq, historical_seq, length)
            if value is not None:
                seq_scores.append((length, value))
        if not seq_scores:
            continue
        # Give the longest available sequence the greatest weight, while retaining
        # short-pattern evidence when old records contain fewer rounds.
        seq_weights = {5: 0.25, 10: 0.33, 15: 0.42}
        seq_similarity = sum(v * seq_weights[n] for n, v in seq_scores) / sum(seq_weights[n] for n, _ in seq_scores)

        context_scores = []
        for key, scale in (("mean", 0.18), ("std", 0.10), ("volatility", 0.12),
                           ("stability", 0.12), ("entropy", 0.10)):
            if current_context.get(key) is None or record.get(key) is None:
                continue
            a, b = _safe_float(current_context.get(key), float("nan")), _safe_float(record.get(key), float("nan"))
            if math.isfinite(a) and math.isfinite(b):
                context_scores.append(clamp(1.0 - abs(a-b) / max(abs(scale), 1e-6), 0.0, 1.0))
        context_similarity = sum(context_scores) / len(context_scores) if context_scores else 0.5
        similarity = clamp(0.80 * seq_similarity + 0.20 * context_similarity, 0.0, 1.0)
        if similarity < 0.20:
            continue

        stamp = _historical_timestamp(record)
        delta_minutes = None
        if stamp is not None:
            # Compare the clock time, not elapsed calendar days: a matching
            # 10:15 context from a previous day is still a 10:15 time-of-day match.
            current_minute = now.hour * 60 + now.minute
            historical_minute = stamp.hour * 60 + stamp.minute
            delta_minutes = abs(current_minute - historical_minute)
            delta_minutes = min(delta_minutes, 1440 - delta_minutes)
        scored.append({"record": record, "similarity": similarity, "delta_minutes": delta_minutes,
                       "outcome": outcome})

    def summarize(items):
        if not items:
            return dict(empty_cohort)
        total = sum(x["similarity"] for x in items)
        if total <= 0:
            return dict(empty_cohort)
        gains = sum(x["similarity"] for x in items if x["outcome"] == "gain")
        target_values, timing_values, delay_values = [], [], []
        for item in items:
            rec, weight = item["record"], item["similarity"]
            target_error = rec.get("target_error")
            if target_error is not None:
                val = _safe_float(target_error, float("nan"))
                if math.isfinite(val): target_values.append((val, weight))
            timing_error = rec.get("timing_error")
            if timing_error is not None:
                val = _safe_float(timing_error, float("nan"))
                if math.isfinite(val): timing_values.append((val, weight))
            actual_delay = rec.get("actual_delay_sec", rec.get("actual_delay", rec.get("delay_seconds")))
            if actual_delay is not None:
                val = _safe_float(actual_delay, float("nan"))
                if math.isfinite(val) and 0 < val <= 1800: delay_values.append((val, weight))
        def wavg(values):
            den = sum(w for _, w in values)
            return round(sum(v*w for v, w in values)/den, 3) if den > 0 else None
        return {
            "samples": len(items),
            "weighted_gain_rate": round(gains/total, 4),
            "avg_target_error": wavg(target_values),
            "avg_timing_error": wavg(timing_values),
            "avg_actual_delay_sec": wavg(delay_values),
            "effective_weight": round(total, 4),
            "mean_similarity": round(total/len(items), 4),
        }

    result["all_similar"] = summarize(scored)
    for window in (5, 15, 30):
        cohort = [x for x in scored if x["delta_minutes"] is not None and x["delta_minutes"] <= window]
        result["time_windows"][str(window)] = summarize(cohort)
    result["status"] = "available" if result["all_similar"]["samples"] else "insufficient_data"
    return result


def build_current_context(cotes, mode, ai_score=0.0) -> dict:
    """
    Construit uniquement le contexte DISPONIBLE AVANT la prédiction.

    IMPORTANT :
    - aucune donnée future
    - aucun actual_delay
    - aucun max_cote_reached
    - aucun résultat de la prédiction actuelle
    """

    if not cotes:
        return {}

    last = list(cotes[-ANALYSE_LAST:])

    m, s = mean_std(last)
    low = consecutive_low(last)
    vol = volatility(last)
    stab = stability_score(last)
    ent = entropy_score(last)
    markov = weighted_markov_next(last)
    bayes = bayes_probability(last)
    peak = compute_peak_interval(last)

    pat = detect_weighted_patterns(
        last,
        pattern_memory,
        SEQ_WEIGHTS
    )

    now = datetime.datetime.now()

    return {
        "mode": mode,
        "sequence": list(last),

        # statistiques pré-prédiction
        "mean": float(m),
        "std": float(s),
        "volatility": float(vol),
        "stability": float(stab),
        "entropy": float(ent),

        # signaux
        "low_count": int(low),
        "low_series": int(low),
        "low_streak": int(low),

        "markov": float(markov),
        "markov_prediction": float(markov),

        "bayes": float(bayes),
        "bayes_probability": float(bayes),

        "peak_interval": (
            float(peak) if peak is not None else None
        ),

        # score courant uniquement à titre informatif.
        # Il NE DOIT PAS déterminer directement le délai.
        "ai_score": float(ai_score),

        "pattern_net": float(pat.get("net", 0.0)),
        "pattern_gain": float(pat.get("raw_gain", 0.0)),
        "pattern_loss": float(pat.get("raw_loss", 0.0)),

        "hour": now.hour,
        "minute": now.minute,
        "timestamp": now.isoformat(),
    }


# Compatibilité avec les anciennes fonctions
def build_current_experience(cotes, mode, ai_score=0.0) -> dict:
    return build_current_context(cotes, mode, ai_score)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _exp_get_low(exp: dict) -> float:
    for key in (
        "low_count",
        "low_series",
        "low_streak",
        "consecutive_low"
    ):
        if exp.get(key) is not None:
            return _safe_float(exp.get(key))
    return 0.0


def _exp_get_markov(exp: dict) -> float:
    for key in (
        "markov",
        "markov_prediction",
        "markov_normalized"
    ):
        if exp.get(key) is not None:
            return _safe_float(exp.get(key))
    return 0.0


def _exp_get_bayes(exp: dict) -> float:
    for key in (
        "bayes_probability",
        "bayes"
    ):
        if exp.get(key) is not None:
            return _safe_float(exp.get(key))
    return 0.0


def _dist_norm(a: float, b: float, scale: float = 1.0) -> float:
    """
    Distance symétrique.

    L'ancien calcul utilisait seulement abs(a) comme référence,
    ce qui rendait la distance dépendante du sens de comparaison.
    """

    a = _safe_float(a)
    b = _safe_float(b)

    denominator = max(
        abs(a),
        abs(b),
        1.0
    )

    return (abs(a - b) / denominator) * scale


def _experience_is_verified(exp: dict) -> bool:
    """
    Une expérience ne devient utilisable comme preuve qu'après feedback.
    """

    if not isinstance(exp, dict):
        return False

    status = exp.get("outcome", exp.get("status"))

    if status == "pending":
        return False

    # Si une expérience possède explicitement un état de vérification
    if exp.get("verified") is False:
        return False

    # Gain/perte = expérience directionnelle vérifiée
    # early/late = expérience temporelle vérifiée
    if status in (
        "gain",
        "loss",
        "perte",
        "early",
        "late"
    ):
        return True

    # Compatibilité avec les anciennes entrées possédant actual_delay
    if exp.get("actual_delay") is not None:
        return True

    return False


def find_similar_experiences(
    current_exp: dict,
    max_results: int = MAX_SIMILAR_EXPERIENCES
) -> list:

    if not experience_memory:
        return []

    current_mode = current_exp.get("mode")

    c_mean = _safe_float(current_exp.get("mean"))
    c_std = _safe_float(current_exp.get("std"))
    c_vol = _safe_float(current_exp.get("volatility"))
    c_stab = _safe_float(current_exp.get("stability"))
    c_ent = _safe_float(current_exp.get("entropy"))

    c_low = _exp_get_low(current_exp)
    c_markov = _exp_get_markov(current_exp)
    c_bayes = _exp_get_bayes(current_exp)
    c_peak = _safe_float(current_exp.get("peak_interval"))

    scored = []

    for exp in experience_memory:

        if not _experience_is_verified(exp):
            continue

        mode_bonus = (
            0.20
            if exp.get("mode") == current_mode
            else 0.0
        )

        e_mean = _safe_float(exp.get("mean"))
        e_std = _safe_float(exp.get("std"))
        e_vol = _safe_float(exp.get("volatility"))
        e_stab = _safe_float(exp.get("stability"))
        e_ent = _safe_float(exp.get("entropy"))
        e_low = _exp_get_low(exp)
        e_markov = _exp_get_markov(exp)
        e_bayes = _exp_get_bayes(exp)
        e_peak = _safe_float(exp.get("peak_interval"))

        dist = (
            _dist_norm(c_mean, e_mean, 0.18)
            + _dist_norm(c_std, e_std, 0.10)
            + _dist_norm(c_vol, e_vol, 0.12)
            + _dist_norm(c_stab, e_stab, 0.12)
            + _dist_norm(c_ent, e_ent, 0.10)

            + (
                abs(c_low - e_low)
                / max(max(c_low, e_low), 1.0)
                * 0.10
            )

            + _dist_norm(
                c_markov,
                e_markov,
                0.10
            )

            + (
                abs(c_bayes - e_bayes)
                * 0.08
            )

            + (
                abs(c_peak - e_peak)
                / max(max(c_peak, e_peak), 1.0)
                * 0.10
            )
        )

        raw_similarity = clamp(
            1.0 / (1.0 + dist)
            + mode_bonus,
            0.0,
            1.0
        )

        reliability = clamp(
            _safe_float(
                exp.get(
                    "reliability",
                    EXP_RELIABILITY_DEFAULT
                ),
                EXP_RELIABILITY_DEFAULT
            ),
            EXP_RELIABILITY_MIN,
            EXP_RELIABILITY_MAX
        )

        age_weight = compute_age_weight(
            exp.get(
                "created_at",
                exp.get("timestamp", "")
            )
        )

        verification_weight = compute_verification_weight(
            int(
                _safe_float(
                    exp.get("verified_count", 0)
                )
            )
        )

        similarity = clamp(
            raw_similarity
            * reliability
            * age_weight
            * verification_weight,
            0.0,
            1.0
        )

        scored.append({
            "similarity": similarity,
            "experience": exp
        })

    scored.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return scored[:max_results]


def aggregate_similar_experiences(
    similar_experiences: list
) -> dict:

    if not similar_experiences:
        return {
            "count": 0,
            "weighted_gain_rate": 0.5,
            "weighted_loss_rate": 0.5,
            "weighted_delay": None,
            "weighted_score": 0.0,
            "confidence": 0.0
        }

    total_weight = 0.0
    gain_weight = 0.0
    loss_weight = 0.0

    delay_sum = 0.0
    delay_weight = 0.0

    score_sum = 0.0
    score_weight = 0.0

    valid_count = 0

    for item in similar_experiences:

        similarity = _safe_float(
            item.get("similarity")
        )

        exp = item.get("experience", {})

        if similarity <= 0:
            continue

        valid_count += 1
        total_weight += similarity

        status = exp.get(
            "status",
            exp.get("outcome")
        )

        if status == "gain":
            gain_weight += similarity

        elif status in (
            "loss",
            "perte"
        ):
            loss_weight += similarity

        observed_delay = exp.get("actual_delay_sec", exp.get("actual_delay"))

        if observed_delay is not None:
            observed_delay = _safe_float(
                observed_delay,
                -1
            )

            if math.isfinite(observed_delay) and 0 < observed_delay <= 1800:
                delay_sum += (
                    observed_delay
                    * similarity
                )
                delay_weight += similarity

        score = exp.get("ai_score")

        if score is not None:
            score = _safe_float(score)
            score_sum += score * similarity
            score_weight += similarity

    if total_weight <= 0:
        return {
            "count": 0,
            "weighted_gain_rate": 0.5,
            "weighted_loss_rate": 0.5,
            "weighted_delay": None,
            "weighted_score": 0.0,
            "confidence": 0.0
        }

    gain_rate = gain_weight / total_weight
    loss_rate = loss_weight / total_weight

    weighted_delay = (
        round(
            delay_sum / delay_weight,
            2
        )
        if delay_weight > 0
        else None
    )

    weighted_score = (
        round(
            score_sum / score_weight,
            4
        )
        if score_weight > 0
        else 0.0
    )

    confidence = clamp(
        (
            min(valid_count, 20)
            / 20.0
        ) * 0.5
        + gain_rate * 0.5,
        0.0,
        1.0
    )

    return {
        "count": valid_count,
        "weighted_gain_rate": round(
            gain_rate,
            4
        ),
        "weighted_loss_rate": round(
            loss_rate,
            4
        ),
        "weighted_delay": weighted_delay,
        "weighted_score": weighted_score,
        "confidence": round(
            confidence,
            4
        )
    }


def retrieve_similar_experiences(
    current_context,
    max_results=MAX_SIMILAR_EXPERIENCES
) -> list:

    return find_similar_experiences(
        current_context,
        max_results=max_results
    )


def get_experience_context(
    cotes,
    mode,
    ai_score=0.0
) -> dict:

    current = build_current_context(
        cotes,
        mode,
        ai_score
    )

    similar = retrieve_similar_experiences(
        current
    )

    aggregate = aggregate_similar_experiences(
        similar
    )

    return {
        "current": current,
        "similar": similar,
        "aggregate": aggregate
    }


# ══════════════════════════════════════════════════════════════════════
# MODULES 7 / 8 / 9
# META DECISION + ADAPTIVE WEIGHTS + ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

def create_meta_model_stats() -> dict:

    model_count = max(
        len(META_MODEL_NAMES),
        1
    )

    return {
        ctx: {
            model: {
                "wins": 0,
                "losses": 0,
                "total": 0,
                "recent_results": [],
                "weight": 1.0 / model_count,
                "reliability": 0.50
            }
            for model in META_MODEL_NAMES
        }
        for ctx in CONTEXTS
    }


def _get_meta_stats(
    context: str,
    model: str
) -> dict:

    global meta_model_stats

    if context not in meta_model_stats:
        meta_model_stats[context] = {}

    if model not in meta_model_stats[context]:

        model_count = max(
            len(META_MODEL_NAMES),
            1
        )

        meta_model_stats[context][model] = {
            "wins": 0,
            "losses": 0,
            "total": 0,
            "recent_results": [],
            "weight": 1.0 / model_count,
            "reliability": 0.50
        }

    return meta_model_stats[context][model]


def compute_meta_model_weight(
    context: str,
    model: str
) -> float:

    stats = _get_meta_stats(
        context,
        model
    )

    wins = int(stats["wins"])
    losses = int(stats["losses"])
    total = int(stats["total"])

    recent = stats.get(
        "recent_results",
        []
    )[-50:]

    historical = smoothed_probability(
        wins,
        losses,
        prior=0.50,
        strength=20
    )

    sample_factor = min(
        1.0,
        total / 100.0
    )

    recent_20 = recent[-20:]

    recent_perf = (
        sum(recent_20)
        / len(recent_20)
        if recent_20
        else 0.50
    )

    combined = (
        historical * 0.50
        + recent_perf
        * (0.30 + sample_factor * 0.20)
        + 0.50
        * max(
            0.0,
            0.20
            - sample_factor * 0.20
        )
    )

    return clamp(
        combined,
        0.10,
        0.95
    )


def normalize_meta_weights(
    context: str
) -> dict:

    raw = {
        model: compute_meta_model_weight(
            context,
            model
        )
        for model in META_MODEL_NAMES
    }

    total = sum(raw.values())

    if total <= 0:

        count = max(
            len(META_MODEL_NAMES),
            1
        )

        return {
            model: 1.0 / count
            for model in META_MODEL_NAMES
        }

    return {
        model: value / total
        for model, value in raw.items()
    }


def compute_model_meta_score(
    model_output: dict,
    context: str,
    model_name: str
) -> float:

    conf = clamp(
        _safe_float(
            model_output.get("confidence")
        ),
        0.0,
        1.0
    )

    samples = max(
        0,
        int(
            _safe_float(
                model_output.get(
                    "samples",
                    0
                )
            )
        )
    )

    uncertainty = clamp(
        _safe_float(
            model_output.get(
                "uncertainty",
                1.0
            ),
            1.0
        ),
        0.0,
        1.0
    )

    stats = _get_meta_stats(
        context,
        model_name
    )

    historical_reliability = (
        stats.get(
            "reliability",
            0.50
        )
    )

    recent = stats.get(
        "recent_results",
        []
    )[-20:]

    recent_perf = (
        sum(recent)
        / len(recent)
        if recent
        else 0.50
    )

    sample_conf = min(
        1.0,
        samples / 200.0
    )

    score = (
        0.30 * historical_reliability
        + 0.25 * conf
        + 0.15 * sample_conf
        + 0.20 * recent_perf
        + 0.10 * (1.0 - uncertainty)
    )

    return clamp(
        score,
        0.0,
        1.0
    )


def meta_decision_engine(
    model_outputs: dict,
    context: str
) -> dict:

    if not model_outputs:

        return {
            "probability": 0.50,
            "meta_scores": {},
            "learned_weights": {},
            "final_weights": {},
            "vote_high": 0.50,
            "vote_low": 0.50,
            "top_model": "neural",
            "top_meta_score": 0.50
        }

    learned_weights = normalize_meta_weights(
        context
    )

    meta_scores = {}
    combined = {}

    for model_name, output in model_outputs.items():

        meta_score = compute_model_meta_score(
            output,
            context,
            model_name
        )

        learned_weight = learned_weights.get(
            model_name,
            1.0 / max(
                len(META_MODEL_NAMES),
                1
            )
        )

        meta_scores[model_name] = meta_score

        combined[model_name] = (
            meta_score
            * learned_weight
        )

    total = sum(combined.values())

    if total <= 0:
        total = 1.0

    final_weights = {
        model: value / total
        for model, value in combined.items()
    }

    final_probability = clamp(
        sum(
            _safe_float(
                model_outputs[model].get(
                    "probability",
                    0.50
                ),
                0.50
            )
            * final_weights[model]
            for model in model_outputs
        ),
        0.01,
        0.99
    )

    vote_high = sum(
        final_weights[model]
        for model in model_outputs
        if _safe_float(
            model_outputs[model].get(
                "probability",
                0.50
            ),
            0.50
        ) >= 0.50
    )

    vote_low = sum(
        final_weights[model]
        for model in model_outputs
        if _safe_float(
            model_outputs[model].get(
                "probability",
                0.50
            ),
            0.50
        ) < 0.50
    )

    ranked = sorted(
        meta_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "probability": round(
            final_probability,
            4
        ),
        "meta_scores": {
            model: round(
                score,
                4
            )
            for model, score in meta_scores.items()
        },
        "learned_weights": {
            model: round(
                weight,
                4
            )
            for model, weight in learned_weights.items()
        },
        "final_weights": {
            model: round(
                final_weights[model],
                4
            )
            for model in final_weights
        },
        "vote_high": round(
            vote_high,
            4
        ),
        "vote_low": round(
            vote_low,
            4
        ),
        "top_model": (
            ranked[0][0]
            if ranked
            else "neural"
        ),
        "top_meta_score": round(
            ranked[0][1],
            4
        ) if ranked else 0.50
    }


def update_meta_model_stats(
    context: str,
    model_outputs: dict,
    success: bool,
    final_prob: float = 0.50
) -> None:

    for model_name, output in model_outputs.items():

        if model_name not in META_MODEL_NAMES:
            continue

        stats = _get_meta_stats(
            context,
            model_name
        )

        model_prob = _safe_float(
            output.get(
                "probability",
                0.50
            ),
            0.50
        )

        model_correct = (
            model_prob >= 0.50
        ) == bool(success)

        stats["total"] += 1
        err = model_prob - float(bool(success))
        stats["brier_sum"] = stats.get("brier_sum", 0.0) + err * err
        p_cal = clamp(model_prob, 1e-6, 1.0 - 1e-6)
        stats["logloss_sum"] = stats.get("logloss_sum", 0.0) - (float(bool(success))*math.log(p_cal) + (1.0-float(bool(success)))*math.log(1.0-p_cal))

        if model_correct:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        stats["recent_results"].append(
            1 if model_correct else 0
        )

        stats["recent_results"] = (
            stats["recent_results"][-50:]
        )

        stats["reliability"] = smoothed_probability(
            stats["wins"],
            stats["losses"],
            prior=0.50,
            strength=20
        )

        stats["weight"] = compute_meta_model_weight(
            context,
            model_name
        )

    logging.info(
        "[META_STATS] ctx=%s success=%s models=%s",
        context,
        success,
        list(model_outputs.keys())
    )


# ══════════════════════════════════════════════════════════════════════
# MODULE 10 — EXPERIENCE RELIABILITY ENGINE
# ══════════════════════════════════════════════════════════════════════

def compute_age_weight(created_at_str: str) -> float:

    try:

        created = datetime.datetime.fromisoformat(
            created_at_str
        )

        age_days = (
            datetime.datetime.now()
            - created
        ).total_seconds() / 86400.0

        return clamp(
            math.exp(
                -age_days / 45.0
            ),
            0.30,
            1.00
        )

    except Exception:
        return 0.70


def compute_verification_weight(
    verified_count: int
) -> float:

    verified_count = max(
        0,
        int(
            _safe_float(
                verified_count
            )
        )
    )

    return clamp(
        0.50
        + min(
            verified_count,
            50
        ) / 100.0,
        0.50,
        1.00
    )


def update_experience_reliability(
    experience_id: str,
    success: bool
) -> None:

    if not experience_id:
        return

    for exp in experience_memory:

        if not isinstance(exp, dict):
            continue

        if exp.get("experience_id") != experience_id:
            continue

        current = _safe_float(
            exp.get(
                "reliability",
                EXP_RELIABILITY_DEFAULT
            ),
            EXP_RELIABILITY_DEFAULT
        )

        if success:

            new_reliability = clamp(
                current
                + EXP_RELIABILITY_GAIN,
                EXP_RELIABILITY_MIN,
                EXP_RELIABILITY_MAX
            )

        else:

            new_reliability = clamp(
                current
                - EXP_RELIABILITY_LOSS,
                EXP_RELIABILITY_MIN,
                EXP_RELIABILITY_MAX
            )

        exp["reliability"] = new_reliability

        exp["verified_count"] = (
            int(
                _safe_float(
                    exp.get(
                        "verified_count",
                        0
                    )
                )
            )
            + 1
        )

        exp["updated_at"] = (
            datetime.datetime.now()
            .isoformat()
        )

        logging.info(
            "[EXP_RELIABILITY] id=%s success=%s reliability=%.3f verified=%s",
            experience_id[:8],
            success,
            new_reliability,
            exp["verified_count"]
        )

        return


# ══════════════════════════════════════════════════════════════════════
# MODULE 5 — ADAPTIVE DELAY ENGINE
# ══════════════════════════════════════════════════════════════════════

def _timing_context_key(
    mode: str,
    cotes: list = None,
    context: str = None,
    pattern_key: str = None
) -> str:
    """
    Construit une clé temporelle robuste par famille de pattern.

    Les cinq dernières cotes sont classées en familles afin que des
    séquences proches partagent leur expérience temporelle.
    """

    if context is None:
        context = detect_market_context(
            cotes or []
        )

    values = list(cotes or [])

    if not values:
        return (
            f"{mode or 'unknown'}"
            f"|{context}"
            f"|UNKNOWN"
        )

    family = []

    for cote in values[-5:]:
        cote = _safe_float(cote, 0.0)

        if cote < 1.30:
            family.append("L")
        elif cote < 2.00:
            family.append("M")
        elif cote < 5.00:
            family.append("H")
        else:
            family.append("X")

    while len(family) < 5:
        family.insert(0, "U")

    family_key = "_".join(family)

    return (
        f"{mode or 'unknown'}"
        f"|{context}"
        f"|{family_key}"
    )


def _ensure_timing_bucket(
    context_key: str,
    delay: int
) -> dict:

    memory = AI_STATE.setdefault(
        "timing_memory",
        {}
    )

    context = memory.setdefault(
        context_key,
        {}
    )

    return context.setdefault(
        str(delay),
        {
            "total": 0,
            "success": 0,
            "loss": 0,
            "early": 0,
            "late": 0,
            "recent_results": [],
            "recent_observed_delays": []
        }
    )


def get_context_delay_score(
    context_key: str,
    delay: int
) -> float:

    if not context_key:
        return 0.50

    bucket = (
        AI_STATE
        .get("timing_memory", {})
        .get(context_key, {})
        .get(str(delay))
    )

    if not bucket:
        return 0.50

    total = int(
        bucket.get("total", 0)
    )

    if total <= 0:
        return 0.50

    return smoothed_probability(
        int(bucket.get("success", 0)),
        int(bucket.get("loss", 0)),
        prior=0.50,
        strength=6
    )


def get_delay_historical_score(
    delay: int
) -> float:

    stats = DELAY_STATS.get(delay)

    if not stats:
        return 0.50

    total = int(
        stats.get("total", 0)
    )

    if total <= 0:
        return 0.50

    return smoothed_probability(
        int(stats.get("success", 0)),
        int(stats.get("loss", 0)),
        prior=0.50,
        strength=10
    )


def recent_delay_performance(
    delay: int
) -> float:

    stats = DELAY_STATS.get(delay)

    if not stats:
        return 0.50

    results = (
        stats
        .get("recent_results", [])
        [-20:]
    )

    if not results:
        return 0.50

    return (
        sum(results)
        / len(results)
    )


def get_delay_evolution_score(
    delay: int
) -> float:

    stats = DELAY_STATS.get(
        delay,
        {}
    )

    recent = stats.get(
        "recent_results",
        []
    )

    if len(recent) < 6:
        return 0.50

    previous = recent[-10:-5]
    current = recent[-5:]

    if not previous or not current:
        return 0.50

    old_score = (
        sum(previous)
        / len(previous)
    )

    new_score = (
        sum(current)
        / len(current)
    )

    evolution = new_score - old_score

    return clamp(
        0.50 + evolution,
        0.0,
        1.0
    )


# ══════════════════════════════════════════════════════════════════════
# TIMING SIMILARITY
# ══════════════════════════════════════════════════════════════════════

def _timing_experience_similarity(
    current: dict,
    exp: dict,
    mode: str
) -> float:
    """
    Similarité temporelle v6.0.

    Compare les statistiques du marché, les cinq dernières cotes,
    l'heure de prédiction, la fiabilité et l'ancienneté de l'expérience.
    """

    if not isinstance(exp, dict):
        return 0.0

    if exp.get("mode") != mode:
        return 0.0

    if not _experience_is_verified(exp):
        return 0.0

    distance = 0.0

    distance += _dist_norm(
        _safe_float(current.get("mean")),
        _safe_float(exp.get("mean")),
        0.22
    )
    distance += _dist_norm(
        _safe_float(current.get("std")),
        _safe_float(exp.get("std")),
        0.14
    )
    distance += _dist_norm(
        _safe_float(current.get("volatility")),
        _safe_float(exp.get("volatility")),
        0.16
    )
    distance += _dist_norm(
        _safe_float(current.get("stability")),
        _safe_float(exp.get("stability")),
        0.14
    )
    distance += _dist_norm(
        _safe_float(current.get("entropy")),
        _safe_float(exp.get("entropy")),
        0.10
    )

    current_low = _safe_float(
        current.get(
            "low_streak",
            current.get("low_series", 0)
        ),
        0.0
    )
    exp_low = _safe_float(
        exp.get("low_streak", 0),
        0.0
    )
    distance += (
        abs(current_low - exp_low)
        / max(max(current_low, exp_low), 1.0)
        * 0.10
    )

    current_bayes = _safe_float(
        current.get(
            "bayes_probability",
            current.get("bayes", 0.5)
        ),
        0.5
    )
    exp_bayes = _safe_float(
        exp.get("bayes_probability", 0.5),
        0.5
    )
    distance += abs(current_bayes - exp_bayes) * 0.08
    distance += _dist_norm(
        _safe_float(current.get("peak_interval")),
        _safe_float(exp.get("peak_interval")),
        0.06
    )

    current_sequence = list(
        current.get("sequence", []) or []
    )
    exp_sequence = list(
        exp.get(
            "cotes",
            exp.get("sequence", [])
        ) or []
    )

    pattern_similarity = 0.50

    if current_sequence and exp_sequence:
        def classify_cote(value):
            value = _safe_float(value, 0.0)
            if value < 1.30:
                return "L"
            if value < 2.00:
                return "M"
            if value < 5.00:
                return "H"
            return "X"

        current_pattern = [
            classify_cote(x)
            for x in current_sequence[-5:]
        ]
        exp_pattern = [
            classify_cote(x)
            for x in exp_sequence[-5:]
        ]
        n = min(len(current_pattern), len(exp_pattern))

        if n > 0:
            matches = sum(
                1
                for i in range(n)
                if current_pattern[-n + i]
                == exp_pattern[-n + i]
            )
            pattern_similarity = matches / n
            distance += (1.0 - pattern_similarity) * 0.18

    current_hour = int(_safe_float(
        current.get("hour", datetime.datetime.now().hour),
        datetime.datetime.now().hour
    ))
    exp_hour = int(_safe_float(exp.get("hour", 0), 0))
    hour_distance = min(
        abs(current_hour - exp_hour),
        24 - abs(current_hour - exp_hour)
    )
    hour_similarity = math.exp(-hour_distance / 4.0)
    distance += (1.0 - hour_similarity) * 0.12

    reliability = clamp(
        _safe_float(
            exp.get(
                "reliability",
                EXP_RELIABILITY_DEFAULT
            ),
            EXP_RELIABILITY_DEFAULT
        ),
        0.10,
        1.0
    )

    age_weight = compute_age_weight(
        exp.get(
            "created_at",
            exp.get(
                "timestamp",
                ""
            )
        )
    )

    base_similarity = 1.0 / (1.0 + distance)
    similarity = (
        base_similarity
        * (
            0.55
            + 0.25 * pattern_similarity
            + 0.20 * hour_similarity
        )
        * reliability
        * age_weight
    )

    return clamp(similarity, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
# TIMING PAR HEURE — v6.0
# ══════════════════════════════════════════════════════════════════════

def get_hour_delay_score(
    mode: str,
    delay: int,
    hour: int = None
) -> float:
    """Score le délai à partir des écarts temporels réellement observés.

    Le statut gain/perte de la cible n'est volontairement pas utilisé comme
    étiquette de réussite temporelle : il appartient au moteur directionnel.
    """
    if hour is None:
        hour = datetime.datetime.now().hour

    weighted_correct = 0.0
    weighted_total = 0.0

    for exp in experience_memory:
        if not isinstance(exp, dict) or exp.get("mode") != mode:
            continue
        if not _experience_is_verified(exp):
            continue

        observed = _safe_float(
            exp.get("actual_delay_sec", exp.get("actual_delay", exp.get("delay_seconds"))),
            -1.0,
        )
        if observed <= 0 or observed > 1800:
            continue

        exp_hour_value = exp.get("hour")
        if exp_hour_value is None:
            try:
                exp_hour_value = datetime.datetime.fromisoformat(
                    str(exp.get("pred_time", exp.get("predicted_datetime", "")))
                ).hour
            except Exception:
                exp_hour_value = None
        if exp_hour_value is None:
            continue

        exp_hour = int(_safe_float(exp_hour_value, hour)) % 24
        hour_distance = min(abs(hour - exp_hour), 24 - abs(hour - exp_hour))
        hour_weight = math.exp(-hour_distance / 3.0)
        reliability = clamp(
            _safe_float(exp.get("reliability", EXP_RELIABILITY_DEFAULT), EXP_RELIABILITY_DEFAULT),
            0.10, 1.0,
        )
        age_weight = compute_age_weight(exp.get("created_at", exp.get("timestamp", exp.get("time", ""))))
        weight = hour_weight * reliability * age_weight
        if weight <= 0:
            continue

        # A timing observation is correct only if the observed event is within
        # ±30 seconds of the selected delay. Target gain/loss is not consulted.
        timing_correct = (_delay_bucket(observed) == int(delay) and abs(observed - _delay_center(int(delay))) <= 30.0)
        weighted_total += weight
        if timing_correct:
            weighted_correct += weight

    if weighted_total <= 0:
        return 0.50
    # Bayesian shrinkage toward neutral avoids one event deciding the hour.
    effective_n = min(20.0, weighted_total)
    raw_rate = weighted_correct / weighted_total
    return clamp((raw_rate * effective_n + 0.5 * 6.0) / (effective_n + 6.0), 0.0, 1.0)


def _delay_center(delay: int) -> float:
    """
    Centre temporel d'une fenêtre.

    Exemple :
    1 min → 90 sec
    2 min → 150 sec
    3 min → 210 sec


    Le nombre de fenêtres vient de SUPPORTED_DELAYS.
    """

    return (
        float(delay) * 60.0
        + 30.0
    )


def _delay_fit(
    observed_delay: float,
    delay: int
) -> float:

    try:
        observed_delay = float(
            observed_delay
        )
    except (
        TypeError,
        ValueError
    ):
        return 0.0

    if observed_delay <= 0:
        return 0.0

    center = _delay_center(
        delay
    )

    # noyau temporel souple.
    # On évite le découpage brutal "1 minute = 60 secondes".
    return math.exp(
        -abs(
            observed_delay
            - center
        ) / 75.0
    )


def get_temporal_bucket_timing_scores(mode: str, now=None) -> dict:
    """Score measured timing observations near the current time of day.

    Uses 5/15/30-minute cohorts, strongest for the nearest cohort. Only
    verified feedback and genuinely measured elapsed times are eligible.
    """
    now = now or datetime.datetime.now()
    current_minute = now.hour * 60 + now.minute
    weighted_fit = {delay: 0.0 for delay in SUPPORTED_DELAYS}
    weighted_support = {delay: 0.0 for delay in SUPPORTED_DELAYS}
    cohort_counts = {"5m": 0, "15m": 0, "30m": 0}
    unique_observations = 0

    for exp in experience_memory:
        if not isinstance(exp, dict) or exp.get("mode") != mode:
            continue
        if not _experience_is_verified(exp):
            continue

        # Prefer prediction time. Legacy hour/minute fields are a safe fallback;
        # feedback/creation timestamps are not substituted for prediction time.
        stamp = (exp.get("pred_time") or exp.get("predicted_datetime")
                 or exp.get("prediction_datetime"))
        try:
            if stamp:
                exp_dt = stamp if isinstance(stamp, datetime.datetime) else datetime.datetime.fromisoformat(str(stamp))
                exp_minute = exp_dt.hour * 60 + exp_dt.minute
            elif exp.get("hour") is not None and exp.get("minute") is not None:
                exp_minute = (int(_safe_float(exp.get("hour"), -1)) % 24) * 60 + int(_safe_float(exp.get("minute"), -1)) % 60
            else:
                continue
        except Exception:
            continue

        raw_distance = abs(current_minute - exp_minute)
        minute_distance = min(raw_distance, 1440 - raw_distance)
        if minute_distance > 30:
            continue
        if minute_distance <= 5:
            temporal_weight, cohort = 1.0, "5m"
        elif minute_distance <= 15:
            temporal_weight, cohort = 0.65, "15m"
        else:
            temporal_weight, cohort = 0.35, "30m"

        observed = _safe_float(
            exp.get("actual_delay_sec", exp.get("actual_delay", exp.get("delay_seconds"))),
            -1.0,
        )
        if not math.isfinite(observed) or observed < 60.0 or observed >= 240.0:
            continue

        reliability = clamp(
            _safe_float(exp.get("reliability", EXP_RELIABILITY_DEFAULT), EXP_RELIABILITY_DEFAULT),
            0.10, 1.0,
        )
        age_weight = compute_age_weight(
            exp.get("created_at", exp.get("timestamp", exp.get("time", "")))
        )
        weight = temporal_weight * reliability * age_weight
        if weight <= 0:
            continue

        unique_observations += 1
        cohort_counts[cohort] += 1
        for delay in SUPPORTED_DELAYS:
            weighted_fit[delay] += _delay_fit(observed, delay) * weight
            weighted_support[delay] += weight

    scores = {}
    for delay in SUPPORTED_DELAYS:
        support = weighted_support[delay]
        raw = weighted_fit[delay] / support if support > 0 else 0.50
        effective_n = min(12.0, support)
        scores[delay] = round((raw * effective_n + 0.50 * 5.0) / (effective_n + 5.0), 4)

    return {
        "scores": scores,
        "samples": unique_observations,
        "support": {str(d): round(weighted_support[d], 3) for d in SUPPORTED_DELAYS},
        "windows_minutes": [5, 15, 30],
        "cohort_counts": cohort_counts,
        "status": "available" if unique_observations >= 4 else "insufficient_samples",
    }


def get_observed_timing_scores(
    mode: str,
    cotes: list,
    context_key: str
) -> dict:

    current = build_current_context(
        cotes,
        mode,
        0.0
    )

    if not current:
        return {
            delay: 0.50
            for delay in SUPPORTED_DELAYS
        }

    candidates = []

    for exp in experience_memory:

        similarity = (
            _timing_experience_similarity(
                current,
                exp,
                mode
            )
        )

        if similarity <= 0:
            continue

        # STRICT: only observed fields are accepted here.
        # delay_sec is a prediction field in the current schema and must
        # never be treated as an observed delay.
        observed = exp.get("actual_delay_sec")
        if observed is None:
            observed = exp.get("actual_delay")
        if observed is None:
            observed = exp.get("delay_seconds")

        observed = _safe_float(
            observed,
            -1
        )

        if observed > 0:
            candidates.append(
                (
                    similarity,
                    observed
                )
            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    candidates = candidates[
        :MAX_SIMILAR_EXPERIENCES
    ]

    scores = {}

    for delay in SUPPORTED_DELAYS:

        weighted = 0.0
        total_weight = 0.0

        # expériences historiques similaires
        for similarity, observed in candidates:

            fit = _delay_fit(
                observed,
                delay
            )

            weighted += (
                fit
                * similarity
            )

            total_weight += similarity

        # observations propres au contexte
        bucket = (
            AI_STATE
            .get("timing_memory", {})
            .get(context_key, {})
            .get(str(delay), {})
        )

        local_observations = bucket.get(
            "recent_observed_delays",
            []
        )

        for observed in local_observations[-20:]:

            observed = _safe_float(
                observed,
                -1
            )

            if observed <= 0:
                continue

            fit = _delay_fit(
                observed,
                delay
            )

            weighted += (
                fit * 1.5
            )

            total_weight += 1.5

        # ----------------------------------------------------------
        # Mémoire horaire complémentaire
        # ----------------------------------------------------------
        current_hour = datetime.datetime.now().hour
        hour_score = get_hour_delay_score(
            mode,
            delay,
            current_hour
        )

        weighted += hour_score * 2.0
        total_weight += 2.0

        scores[delay] = round(
            weighted / total_weight
            if total_weight > 0
            else 0.50,
            4
        )

    # Les cohortes temporelles apportent un signal complémentaire limité à 15 %.
    # En dessous de quatre observations mesurées, elles restent purement diagnostiques.
    try:
        temporal = get_temporal_bucket_timing_scores(mode)
        result_temporal = temporal.get("scores", {})
        if temporal.get("status") == "available":
            temporal_weight = min(0.15, 0.03 + 0.01 * max(0, int(temporal.get("samples", 0)) - 4))
            for delay in SUPPORTED_DELAYS:
                scores[delay] = round(
                    (1.0 - temporal_weight) * scores[delay]
                    + temporal_weight * _safe_float(result_temporal.get(delay, 0.50), 0.50),
                    4,
                )
            logging.info(
                "[TEMPORAL BUCKET TIMING] mode=%s samples=%s weight=%.3f scores=%s",
                mode, temporal.get("samples"), temporal_weight, result_temporal,
            )
    except Exception as temporal_exc:
        logging.warning("[TEMPORAL BUCKET TIMING] skipped: %s", temporal_exc)

    return scores


# État explicite du modèle temporel : un fallback uniforme n'est jamais
# présenté comme une prédiction apprise.
DELAY_MODEL_STATUS = "untrained"

# ══════════════════════════════════════════════════════════════════════
# LEARNED DELAY PREDICTION
# ══════════════════════════════════════════════════════════════════════

def get_learned_delay_predictions(
    last: list,
    mode: str,
    context_key: str
) -> dict:
    global DELAY_MODEL_STATUS
    """
    Produit les probabilités des fenêtres temporelles.

    PRIORITÉ :
        1. modèle DelayNet s'il est entraîné
        2. mémoire temporelle vérifiée
        3. historique global

    Les heuristiques ne choisissent plus directement la minute.
    """

    delays = list(
        SUPPORTED_DELAYS
    )

    if not delays:
        return {}

    # --------------------------------------------------------------
    # 1. Construction des features disponibles AVANT prédiction
    # --------------------------------------------------------------

    m, s = mean_std(last)
    low = consecutive_low(last)
    vol = volatility(last)
    peak = compute_peak_interval(last) or 0.0

    pat = detect_weighted_patterns(
        last,
        pattern_memory,
        SEQ_WEIGHTS
    )

    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute

    # Le DelayNet de la partie 1 doit utiliser uniquement ces
    # informations pré-prédiction.
    features = build_prediction_features(
        consecutive_low_value=low,
        volatility_value=vol,
        std_value=s,
        mean_value=m,

        # IMPORTANT :
        # aucun score cible futur ici.
        predicted_score=0.0,
        predicted_target=0.0,

        predicted_hour=hour,
        predicted_minute=minute,

        peak_interval=peak,
        sequence_length=len(last),

        pattern_gain=pat.get(
            "raw_gain",
            0.0
        ),
        pattern_loss=pat.get(
            "raw_loss",
            0.0
        )
    )

    # --------------------------------------------------------------
    # 2. Prédiction du DelayNet
    # --------------------------------------------------------------

    learned = {}

    try:

        # compatibilité avec plusieurs implémentations
        if "delay_engine" in globals() and delay_engine is not None:

            prediction = None

            if hasattr(
                delay_engine,
                "predict_distribution"
            ):
                prediction = (
                    delay_engine.predict_distribution(
                        features
                    )
                )

            elif hasattr(
                delay_engine,
                "predict"
            ):
                prediction = (
                    delay_engine.predict(
                        features
                    )
                )

            if isinstance(
                prediction,
                dict
            ):

                for delay in delays:
                    value = prediction.get(
                        delay,
                        prediction.get(
                            str(delay),
                            0.50
                        )
                    )

                    learned[delay] = clamp(
                        _safe_float(
                            value,
                            0.50
                        ),
                        0.01,
                        0.99
                    )

        # fallback vers l'objet/model utilisé dans la partie 1
        if not learned and "DelayNet" in globals():

            # On ne force pas ici une architecture différente.
            # Cette branche reste volontairement compatible.
            pass

    except Exception as exc:

        logging.warning(
            "[DELAY_NET] prediction failed: %s",
            exc
        )

    # Si le modèle n'est pas encore entraîné, utiliser un prior uniforme
    # explicitement marqué comme non appris.
    if not learned:
        DELAY_MODEL_STATUS = "untrained"
        learned = {delay: 0.50 for delay in delays}
    else:
        DELAY_MODEL_STATUS = "trained"

    # --------------------------------------------------------------
    # 3. Normalisation
    # --------------------------------------------------------------

    total = sum(
        learned.values()
    )

    if total > 0:

        learned = {
            delay: value / total
            for delay, value in learned.items()
        }

    # Calibration historique intégrée au moteur V6.
    try:
        if V6_TIMING_CALIBRATION.get("status") == "trained":
            learned = _v6_calibrate_delay_distribution(learned)
            DELAY_MODEL_STATUS = "trained_calibrated"
    except Exception as exc:
        logging.warning("[V6 TIMING CALIBRATION] fallback: %s", exc)

    return learned


# ══════════════════════════════════════════════════════════════════════
# DELAY COMPETITION — CORRECTED
# ══════════════════════════════════════════════════════════════════════

def v6_learn_timing_fusion_weights(records=None):
    """Learn bounded timing-source weights from resolved historical feedback."""
    names=("temporal","observed","context","recent","evolution","pattern","pattern_timing","historical","hour")
    sums={n:0.0 for n in names}; counts={n:0.0 for n in names}
    for rec in (records if isinstance(records,list) else feedback_history):
        if not isinstance(rec,dict): continue
        actual=safe_float(rec.get("actual_delay_sec"),float("nan"))
        if not math.isfinite(actual) or actual<=0: continue
        bucket=_v6_bucket(actual)
        if bucket not in SUPPORTED_DELAYS: continue
        scores=rec.get("delay_scores") or {}
        for name in names:
            src=scores.get(str(bucket),scores.get(bucket,{}))
            if not isinstance(src,dict): continue
            pred=_v6_safe_probability(src.get(name),0.5)
            err=(pred-1.0)**2
            for d in SUPPORTED_DELAYS:
                if d==bucket: continue
                other=scores.get(str(d),scores.get(d,{}))
                if isinstance(other,dict): err += _v6_safe_probability(other.get(name),0.5)**2/max(len(SUPPORTED_DELAYS)-1,1)
            w=v6_recency_weight(rec.get("time") or rec.get("created_at") or "")
            sums[name]+=w*err; counts[name]+=w
    raw={n:(1.0/max(sums[n]/counts[n],0.02) if counts[n]>0 else 1.0) for n in names}
    total=sum(raw.values()) or 1.0
    bounded={k:clamp(v/total,V6_FUSION_MIN_WEIGHT,V6_FUSION_MAX_WEIGHT) for k,v in raw.items()}
    z=sum(bounded.values()) or 1.0
    return {k:v/z for k,v in bounded.items()}


def v6_magnitude_distribution(last, mode):
    """Empirical magnitude distribution independent from direction and timing."""
    bins={"2-5x":(2.0,5.0),"5-10x":(5.0,10.0),"10-20x":(10.0,20.0),"20x+":(20.0,float("inf"))}
    counts={k:0.0 for k in bins}
    for rec in feedback_history:
        if not isinstance(rec,dict) or rec.get("status") not in ("gain","perte","loss","early","late"): continue
        if mode and rec.get("mode") not in (None,mode): continue
        value=safe_float(rec.get("max_cote"),float("nan"))
        if not math.isfinite(value) or value<2.0: continue
        w=v6_recency_weight(rec.get("time") or rec.get("created_at") or "",45.0)
        for name,(lo,hi) in bins.items():
            if lo<=value<hi: counts[name]+=w; break
    total=sum(counts.values())
    if total<V6_MAGNITUDE_MIN_SAMPLES:
        probs={k:1.0/len(bins) for k in bins}; status="prior"
    else:
        probs={k:v/total for k,v in counts.items()}; status="empirical"
    entropy=v6_timing_entropy(probs)
    confidence=clamp((1.0-entropy)*0.7+min(total/100.0,1.0)*0.3,0,1)
    return {"probabilities":{k:round(v,4) for k,v in probs.items()},"confidence":round(confidence,4),"status":status,"samples":int(total)}



def v6_target_bias_correction(mode):
    """Correction empirique bornée du biais de cible à partir des feedbacks."""
    ratios=[]
    for rec in feedback_history:
        if not isinstance(rec,dict) or rec.get("status") not in ("gain","perte","loss"):
            continue
        if mode and rec.get("mode") not in (None,mode):
            continue
        actual=safe_float(rec.get("max_cote"),float("nan"))
        pred=safe_float((rec.get("prediction") or {}).get("target"),float("nan"))
        if not math.isfinite(actual) or not math.isfinite(pred) or actual<=0 or pred<=0:
            continue
        ratios.append((clamp(actual/pred,0.50,1.50),v6_recency_weight(rec.get("time") or rec.get("created_at") or "",45.0)))
    if len(ratios)<10:
        return {"factor":1.0,"samples":len(ratios),"confidence":0.0}
    ratios.sort(key=lambda x:x[0]); total=sum(w for _,w in ratios); acc=0.0; median=1.0
    for value,weight in ratios:
        acc+=weight
        if acc>=total*.50: median=value; break
    factor=clamp(1.0+0.35*(median-1.0),0.88,1.12)
    confidence=clamp(min(len(ratios)/50.0,1.0)*.65+(1.0-abs(median-1.0)/.5)*.35,0.0,1.0)
    return {"factor":factor,"samples":len(ratios),"confidence":confidence,"median_ratio":median}

def compute_delay_competition(
    delay_predictions: dict,
    pattern_scores: dict = None,
    recent_context: dict = None,
    observed_scores: dict = None,
    pattern_timing_scores: dict = None
) -> dict:

    results = {}

    context_key = (
        recent_context or {}
    ).get(
        "timing_context_key"
    )
    mode = (
        recent_context or {}
    ).get(
        "mode",
        "unknown"
    )

    for delay in SUPPORTED_DELAYS:

        # ----------------------------------------------------------
        # Evidence 1 : modèle temporel appris
        # ----------------------------------------------------------

        temporal_score = clamp(
            _safe_float(
                delay_predictions.get(
                    delay,
                    0.0
                )
            ),
            0.0,
            1.0
        )

        # ----------------------------------------------------------
        # Evidence 2 : historique global
        # ----------------------------------------------------------

        historical_score = (
            get_delay_historical_score(
                delay
            )
        )

        # ----------------------------------------------------------
        # Evidence 3 : performance récente
        # ----------------------------------------------------------

        recent_score = (
            recent_delay_performance(
                delay
            )
        )

        # ----------------------------------------------------------
        # Evidence 4 : mémoire de timing observé
        # ----------------------------------------------------------

        observed_score = clamp(
            _safe_float(
                (observed_scores or {}).get(
                    delay,
                    0.50
                ),
                0.50
            ),
            0.0,
            1.0
        )

        # ----------------------------------------------------------
        # Evidence 5 : évolution
        # ----------------------------------------------------------

        evolution_score = (
            get_delay_evolution_score(
                delay
            )
        )

        # ----------------------------------------------------------
        # Evidence 6 : contexte exact
        # ----------------------------------------------------------

        context_bucket = {}

        if context_key:

            context_bucket = (
                AI_STATE
                .get(
                    "timing_memory",
                    {}
                )
                .get(
                    context_key,
                    {}
                )
                .get(
                    str(delay),
                    {}
                )
            )

        context_samples = int(
            context_bucket.get(
                "total",
                0
            )
        )

        context_score = (
            get_context_delay_score(
                context_key,
                delay
            )
            if context_key
            else 0.50
        )

        # ----------------------------------------------------------
        # Plus le même contexte possède de feedback,
        # plus son expérience remplace le prior global.
        # ----------------------------------------------------------

        context_weight = min(
            0.50,
            context_samples / 20.0
        )

        learned_context = (
            context_score
            * context_weight
            + historical_score
            * (1.0 - context_weight)
        )

        pattern_score = clamp(
            _safe_float(
                (pattern_scores or {}).get(
                    delay,
                    0.50
                ),
                0.50
            ),
            0.0,
            1.0
        )

        # ----------------------------------------------------------
        # Evidence 7 : historique par heure
        # ----------------------------------------------------------
        current_hour = datetime.datetime.now().hour
        hour_score = get_hour_delay_score(
            mode,
            delay,
            current_hour
        )

        # ----------------------------------------------------------
        # Evidence 8 : mémoire PATTERN → DÉLAI
        # ----------------------------------------------------------
        pattern_timing_score = clamp(
            _safe_float(
                (pattern_timing_scores or {}).get(delay, 0.50),
                0.50
            ),
            0.0,
            1.0
        )
        pattern_sequence = (recent_context or {}).get("pattern_sequence", [])
        pattern_info = get_pattern_timing_score(pattern_sequence, mode, delay) if pattern_sequence else {"samples": 0, "reliability": 0.0}
        pattern_samples = int(_safe_float(pattern_info.get("samples", 0), 0))
        pattern_reliability = clamp(_safe_float(pattern_info.get("reliability", 0.0), 0.0), 0.0, 1.0)
        # A sparse sub-pattern can inform timing only in proportion to its
        # measured reliability; it cannot dominate merely because it matches.
        pattern_timing_score = 0.50 + (pattern_timing_score - 0.50) * pattern_reliability
        global_samples = int(_safe_float((DELAY_STATS.get(delay) or {}).get("total", 0), 0))
        similar_hour_samples = 0
        try:
            now_hour = datetime.datetime.now().hour
            for exp in experience_memory:
                if not isinstance(exp, dict) or exp.get("mode") != mode or not _experience_is_verified(exp):
                    continue
                observed = _safe_float(exp.get("actual_delay_sec", exp.get("actual_delay")), -1.0)
                if observed <= 0 or _delay_bucket(observed) != delay:
                    continue
                exp_hour = exp.get("hour")
                if exp_hour is None:
                    try:
                        exp_hour = datetime.datetime.fromisoformat(str(exp.get("pred_time", ""))).hour
                    except Exception:
                        exp_hour = None
                if exp_hour is not None:
                    gap = abs(now_hour - int(_safe_float(exp_hour, now_hour)))
                    gap = min(gap, 24-gap)
                    if gap <= 3:
                        similar_hour_samples += 1
        except Exception:
            similar_hour_samples = 0
        # Max, not sum: the same feedback may exist in several derived memories.
        evidence_samples = max(context_samples, pattern_samples, global_samples, similar_hour_samples)

        # ----------------------------------------------------------
        # FUSION
        #
        # Le modèle appris et le timing réellement observé
        # sont maintenant prioritaires.
        #
        # Aucune utilisation directe de ai_score.
        # ----------------------------------------------------------

        fusion = v6_learn_timing_fusion_weights()
        source_values = {"temporal":temporal_score,"observed":observed_score,"context":learned_context,"recent":recent_score,"evolution":evolution_score,"pattern":pattern_score,"pattern_timing":pattern_timing_score,"historical":historical_score,"hour":hour_score}
        final_score = sum(fusion[k]*source_values[k] for k in source_values)

        results[delay] = {
            "temporal": round(
                temporal_score,
                4
            ),
            "historical": round(
                historical_score,
                4
            ),
            "context": round(
                context_score,
                4
            ),
            "recent": round(
                recent_score,
                4
            ),
            "observed": round(
                observed_score,
                4
            ),
            "pattern": round(
                pattern_score,
                4
            ),
            "evolution": round(
                evolution_score,
                4
            ),
            "hour": round(
                hour_score,
                4
            ),
            "pattern_timing": round(
                pattern_timing_score,
                4
            ),
            "context_samples": context_samples,
            "pattern_samples": pattern_samples,
            "pattern_reliability": round(pattern_reliability, 4),
            "global_samples": global_samples,
            "similar_hour_samples": similar_hour_samples,
            "evidence_samples": evidence_samples,
            "final_score": round(
                final_score,
                4
            )
        }

    return results


def _timing_selection_probabilities(delay_scores: dict) -> tuple:
    """Distribution prudente, tenant compte du soutien de chaque délai."""
    available = [d for d in SUPPORTED_DELAYS if d in (delay_scores or {})]
    if not available:
        return {d: 1.0 / len(SUPPORTED_DELAYS) for d in SUPPORTED_DELAYS}, 0.0, 0, 0

    values = {d: clamp(_safe_float((delay_scores[d] or {}).get("final_score"), 0.5), 0.0, 1.0) for d in available}
    support_by_delay = {
        d: max(0, int(_safe_float((delay_scores[d] or {}).get("evidence_samples", 0), 0)))
        for d in available
    }
    spread = max(values.values()) - min(values.values()) if values else 0.0
    total_support = sum(support_by_delay.values())
    min_support = min(support_by_delay.values()) if support_by_delay else 0

    mx = max(values.values())
    exps = {d: math.exp((values[d] - mx) / 0.16) for d in available}
    exp_total = sum(exps.values()) or 1.0
    learned = {d: exps[d] / exp_total for d in available}

    # If even one alternative is poorly observed, the apparent winner may be
    # a data-coverage artifact. Shrink more strongly toward neutral in that case.
    if min_support < 3:
        shrink = 0.82
    elif min_support < 8:
        shrink = 0.62
    elif spread < 0.035:
        shrink = 0.70
    elif spread < 0.08:
        shrink = 0.30
    else:
        shrink = 0.08

    uniform = 1.0 / len(available)
    probs = {d: (1.0 - shrink) * learned[d] + shrink * uniform for d in available}
    for d in SUPPORTED_DELAYS:
        probs.setdefault(d, 1.0 / len(SUPPORTED_DELAYS))
    norm = sum(probs.values()) or 1.0
    probs = {d: probs[d] / norm for d in SUPPORTED_DELAYS}
    return probs, spread, total_support, min_support


def select_best_delay(delay_scores: dict) -> int:
    """Choose 1/2/3 from temporal evidence, with uncertainty-aware exploration.

    When evidence is sparse or scores are nearly tied, sample from the softened
    distribution instead of always taking argmax. With stronger evidence, use
    the best-supported delay. This avoids deterministic score lock-in without
    forcing a different minute when the evidence is genuinely decisive.
    """
    probs, spread, total_support, min_support = _timing_selection_probabilities(delay_scores)
    weak_evidence = min_support < 8 or spread < 0.035
    if weak_evidence:
        delays = list(SUPPORTED_DELAYS)
        weights = [max(0.001, probs[d]) for d in delays]
        return int(random.choices(delays, weights=weights, k=1)[0])
    return int(max(probs, key=probs.get))


def live_timing_decision(delay_scores: dict, context_key: str = "") -> tuple:
    """Master timing decision: independently competes 1/2/3 at each prediction.

    The directional ai_score is intentionally not used. Sparse or tied evidence
    uses uncertainty-aware exploration; supported differences remain decisive.
    """
    if set(SUPPORTED_DELAYS) != {1, 2, 3}:
        raise RuntimeError(f"Timing configuration must be exactly 1/2/3, got {SUPPORTED_DELAYS}")
    probs, spread, total_support, min_support = _timing_selection_probabilities(delay_scores)
    selected = select_best_delay(delay_scores)

    # Keep a compact audit trail so persistent minute concentration is visible.
    audit = AI_STATE.setdefault("timing_selection_audit", {})
    recent = audit.setdefault("recent", [])
    recent.append({
        "delay": selected,
        "time": datetime.datetime.now().isoformat(),
        "min_support": min_support,
        "spread": round(spread, 4),
    })
    audit["recent"] = recent[-120:]
    recent30 = audit["recent"][-30:]
    if len(recent30) >= 20:
        counts = {d: sum(1 for item in recent30 if item.get("delay") == d) for d in SUPPORTED_DELAYS}
        dominant = max(counts, key=counts.get)
        if counts[dominant] / len(recent30) >= 0.85 and min_support < 8:
            logging.warning("[TIMING BIAS AUDIT] concentration=%s/%s sur %s min avec support incomplet=%s; exploration maintenue.", counts[dominant], len(recent30), dominant, min_support)

    logging.info(
        "[LIVE TIMING] context=%s support_total=%s support_min=%s spread=%.4f scores=%s probabilities=%s selected=%s",
        context_key,
        total_support,
        min_support,
        spread,
        {d: round(_safe_float((delay_scores.get(d) or {}).get("final_score"), 0.5), 4) for d in SUPPORTED_DELAYS},
        {d: round(probs[d], 4) for d in probs},
        selected,
    )
    return selected, probs


# ══════════════════════════════════════════════════════════════════════
# DELAY FEEDBACK
# ══════════════════════════════════════════════════════════════════════

def update_delay_stats(
    delay: int,
    success: bool,
    context_key: str = None,
    timing_status: str = None,
    observed_delay: float = None
) -> None:

    delay = int(delay)

    if delay not in DELAY_STATS:
        return

    stats = DELAY_STATS[delay]

    stats["total"] += 1

    if success:
        stats["success"] += 1
    else:
        stats["loss"] += 1

    stats.setdefault(
        "recent_results",
        []
    ).append(
        1 if success else 0
    )

    stats["recent_results"] = (
        stats["recent_results"][-50:]
    )

    total = (
        stats["success"]
        + stats["loss"]
    )

    stats["avg_score"] = (
        round(
            stats["success"]
            / total,
            4
        )
        if total > 0
        else 0.0
    )

    if context_key:

        bucket = _ensure_timing_bucket(
            context_key,
            delay
        )

        bucket["total"] += 1

        if success:
            bucket["success"] += 1
        else:
            bucket["loss"] += 1

        if timing_status == "early":
            bucket["early"] += 1

        elif timing_status == "late":
            bucket["late"] += 1

        bucket["recent_results"].append(
            1 if success else 0
        )

        bucket["recent_results"] = (
            bucket["recent_results"][-50:]
        )

        if observed_delay is not None:

            observed_delay = _safe_float(
                observed_delay,
                -1
            )

            if observed_delay > 0:

                bucket[
                    "recent_observed_delays"
                ].append(
                    round(
                        observed_delay,
                        2
                    )
                )

                bucket[
                    "recent_observed_delays"
                ] = bucket[
                    "recent_observed_delays"
                ][-50:]

    logging.info(
        "[DELAY_STATS] delay=%s success=%s total=%s win_rate=%.3f",
        delay,
        success,
        stats["total"],
        stats["avg_score"]
    )


def _apply_timing_feedback(selected, status, actual_delay=None, context_key=None):
    """Apply one timing result to the selected window, never a second window."""
    try:
        selected = int(selected)
    except (TypeError, ValueError):
        return False
    if selected not in SUPPORTED_DELAYS:
        selected = min(SUPPORTED_DELAYS, key=lambda d: abs(d - selected))

    status = str(status or "").lower().strip()
    if status in ("gain", "perte", "loss"):
        actual_delay = _safe_float(actual_delay, -1.0)
        if not 0 < actual_delay <= 1800:
            return False
        observed = _delay_bucket(actual_delay)
        timing_correct = (
            observed == selected
            and abs(actual_delay - _delay_center(selected)) <= 30.0
        )
        update_delay_stats(
            selected,
            timing_correct,
            context_key=context_key,
            observed_delay=actual_delay,
        )
        return True

    if status in ("early", "late"):
        # Qualitative correction: the selected window was wrong, but no
        # neighboring window is credited without an actual prediction.
        update_delay_stats(
            selected,
            False,
            context_key=context_key,
            timing_status=status,
        )
        return True

    return False


def _has_observed_feedback_time(feedback):
    """True when this feedback contains a valid user-observed elapsed time."""
    if not isinstance(feedback, dict):
        return False
    actual_delay = _safe_float(feedback.get("actual_delay_sec"), -1.0)
    source = feedback.get("actual_delay_source")
    if not source and feedback.get("actual_datetime"):
        source = "observed_datetime"
    return (
        source == "observed_datetime"
        and 0 < actual_delay <= 1800
    )


def _nearest_delay(
    actual_delay: float
) -> int:

    actual_delay = _safe_float(
        actual_delay,
        -1
    )

    if actual_delay <= 0:
        return (
            SUPPORTED_DELAYS[0]
            if SUPPORTED_DELAYS
            else 2
        )

    # Le feedback -1/+1 est exprimé en minutes entières. Utiliser
    # directement l'arrondi évite une égalité artificielle entre les centres
    # de deux fenêtres (120s doit représenter 2 minutes, pas 1 minute).
    rounded_minutes = int(round(actual_delay / 60.0))
    return min(
        SUPPORTED_DELAYS,
        key=lambda d: abs(d - rounded_minutes)
    )


def update_timing_feedback(
    prediction: dict,
    status: str,
    actual_delay: float = None
) -> None:
    if not prediction:
        return
    result = prediction.get("result", prediction)
    selected = int(result.get("delay_min", 2))
    if selected not in SUPPORTED_DELAYS:
        selected = min(SUPPORTED_DELAYS, key=lambda d: abs(d - selected))
    context_key = result.get("timing_context_key") or prediction.get("timing_context_key")
    status = str(status or "").lower().strip()

    # Directional outcome alone is not timing evidence. A measured timestamp
    # scores only the window that was actually selected for that prediction.
    _apply_timing_feedback(selected, status, actual_delay, context_key)


def rebuild_delay_stats_from_feedback():
    """Rebuild global and contextual delay scores from one canonical record per feedback."""
    global DELAY_STATS

    DELAY_STATS = {
        delay: {
            "total": 0,
            "success": 0,
            "loss": 0,
            "recent_results": [],
            "avg_score": 0.0,
        }
        for delay in SUPPORTED_DELAYS
    }
    if isinstance(AI_STATE, dict):
        AI_STATE["timing_memory"] = {}

    rebuilt = 0
    for feedback in feedback_history if isinstance(feedback_history, list) else []:
        if not isinstance(feedback, dict):
            continue
        status = str(
            feedback.get("learning_status") or feedback.get("status") or ""
        ).lower().strip()
        if status not in ("gain", "perte", "loss", "early", "late"):
            continue

        prediction = feedback.get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        selected = prediction.get(
            "delay_min",
            feedback.get("predicted_delay", feedback.get("delay_min")),
        )
        if selected is None:
            selected = round(
                _safe_float(prediction.get("delay_sec"), 120.0) / 60.0
            )
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            selected = 2
        selected = max(min(SUPPORTED_DELAYS), min(max(SUPPORTED_DELAYS), selected))

        actual_delay = _safe_float(feedback.get("actual_delay_sec"), -1.0)
        measured = _has_observed_feedback_time(feedback)
        if status in ("gain", "perte", "loss") and not measured:
            continue

        context_key = (
            feedback.get("timing_context_key")
            or prediction.get("timing_context_key")
        )
        if context_key is None:
            cotes = (
                feedback.get("cotes")
                or prediction.get("cotes")
                or prediction.get("last_cotes")
            )
            if cotes:
                try:
                    context_key = _timing_context_key(
                        feedback.get("mode", prediction.get("mode")),
                        cotes,
                    )
                except Exception:
                    context_key = None

        if _apply_timing_feedback(
            selected,
            status,
            actual_delay if measured else None,
            context_key,
        ):
            rebuilt += 1

    if isinstance(AI_STATE, dict):
        AI_STATE["DELAY_STATS"] = {
            str(key): value for key, value in DELAY_STATS.items()
        }
    return rebuilt


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# V6 — ENSEMBLE INTELLIGENCE / CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════
V6_ENSEMBLE_MIN_SAMPLES = 12
V6_ENSEMBLE_MAX_HISTORY = 5000
V6_ENSEMBLE_RECENCY_HALF_LIFE = 30.0

def _v6_norm_signal(value, default=0.5):
    try: x=float(value)
    except Exception: return float(default)
    if not math.isfinite(x): return float(default)
    if x > 1.0: x /= 100.0
    return clamp(x, 0.0, 1.0)

def _v6_sigmoid(x):
    try:
        x=float(x)
        if x >= 0:
            z=math.exp(-min(x,40.0)); return 1.0/(1.0+z)
        z=math.exp(max(x,-40.0)); return z/(1.0+z)
    except Exception: return 0.5

def _v6_directional_signals(cotes, mode, pattern_result=None, memory_result=None):
    last=list(cotes or [])[-ANALYSE_LAST:]
    if not last: return {}
    m,sd=mean_std(last); sd=max(sd,0.20)
    low=max(0.0,safe_float(consecutive_low(last),0.0))
    markov_raw=safe_float(weighted_markov_next(last),m)
    bayes_raw=safe_float(bayes_probability(last),0.5)
    entropy=_v6_norm_signal(entropy_score(last),0.5)
    stability=_v6_norm_signal(stability_score(last),0.5)
    pat=safe_float((pattern_result or {}).get('net'),0.0)
    mem=_v6_norm_signal((memory_result or {}).get('memory_score'),0.5)
    mem_conf=_v6_norm_signal((memory_result or {}).get('confidence'),0.0)
    vol=max(0.0,safe_float(volatility(last),0.0))
    markov=_v6_sigmoid((markov_raw-m)/sd)
    bayes=bayes_raw if 0<=bayes_raw<=1 else _v6_sigmoid((bayes_raw-m)/sd)
    volatility_signal=_v6_sigmoid((vol-.65)*3.0)
    return {
      'markov':clamp(markov,0,1),'bayes':clamp(bayes,0,1),
      'pattern':clamp(.5+pat,0,1),'memory':mem,'memory_confidence':mem_conf,
      'stability':stability,'low_series':_v6_sigmoid((low-1.5)*.9),
      'magnitude':_v6_sigmoid((m+.5*sd-2.0)/1.8),
      'context':clamp(.50*stability+.30*(1-entropy)+.20*(1-volatility_signal),0,1),
    }

def _v6_feedback_engine_weights(mode):
    base={'markov':.13,'bayes':.12,'pattern':.15,'memory':.15,'stability':.12,'low_series':.09,'magnitude':.13,'context':.11}
    samples=[]
    for fb in (feedback_history or []):
        if not isinstance(fb,dict) or str(fb.get('mode',''))!=str(mode): continue
        outcome=str(fb.get('outcome',fb.get('result',''))).lower()
        if outcome not in ('gain','perte','loss','win','lose'): continue
        outputs=fb.get('ensemble_signals') or fb.get('model_outputs')
        if isinstance(outputs,dict): samples.append((outputs,1.0 if outcome in ('gain','win') else 0.0,fb.get('timestamp',fb.get('created_at',0))))
    if len(samples)<V6_ENSEMBLE_MIN_SAMPLES: return base,len(samples)
    now=time.time(); quality={k:[] for k in base}
    for outputs,target,stamp in samples[-V6_ENSEMBLE_MAX_HISTORY:]:
        try: age=max(0,(now-float(stamp))/86400) if stamp else 0
        except Exception: age=0
        rec=math.exp(-math.log(2)*age/V6_ENSEMBLE_RECENCY_HALF_LIFE)
        for k in base:
            if k in outputs: quality[k].append((1-abs(_v6_norm_signal(outputs[k],.5)-target))*rec)
    weights={}
    for k,v in base.items():
        q=sum(quality[k])/len(quality[k]) if quality[k] else .5
        weights[k]=clamp(v*(.70+.90*q),.04,.32)
    total=sum(weights.values()) or 1; return {k:v/total for k,v in weights.items()},len(samples)

def v6_ensemble_decision(cotes, mode, ai_score, pattern_result=None, memory_result=None):
    signals=_v6_directional_signals(cotes,mode,pattern_result,memory_result)
    if not signals: return {'score':_v6_norm_signal(ai_score),'agreement':0,'evidence':0,'novelty':1,'uncertainty':1,'signals':{},'weights':{}}
    weights,n=_v6_feedback_engine_weights(mode); engines=dict(signals); engines['ai_score']=_v6_norm_signal(ai_score)
    weights['ai_score']=.15; total=sum(weights.values()) or 1; weights={k:v/total for k,v in weights.items()}
    fused=sum(weights.get(k,0)*_v6_norm_signal(v,.5) for k,v in engines.items())
    vals=[_v6_norm_signal(v,.5) for v in engines.values()]
    dispersion=math.sqrt(sum((v-fused)**2 for v in vals)/max(1,len(vals)))
    agreement=clamp(1-dispersion*2.4,0,1)
    evidence=clamp(.45*agreement+.20*signals.get('memory_confidence',0)+.20*signals.get('context',.5)+.15*min(1,len(cotes or [])/15),0,1)
    novelty=clamp(1-signals.get('context',.5),0,1)
    uncertainty=clamp(.45*(1-agreement)+.30*(1-evidence)+.25*novelty,0,1)
    return {'score':round(clamp(fused,0,1),6),'agreement':round(agreement,6),'evidence':round(evidence,6),'novelty':round(novelty,6),'uncertainty':round(uncertainty,6),'signals':{k:round(_v6_norm_signal(v,.5),6) for k,v in engines.items()},'weights':{k:round(v,6) for k,v in weights.items()},'feedback_samples':n}


def v7_independent_final_decision(result, memory_result=None):
    """Independent, bounded final consistency check for the existing ensemble.

    This is a last quality-control layer, not a new predictor. It only uses
    pre-outcome signals already attached to the current prediction and
    completed historical cohorts. Its influence is deliberately capped.
    """
    result = dict(result or {})
    current = clamp(safe_float(result.get("score"), 0.5), 0.0, 1.0)
    signals = result.get("ensemble_signals") or {}
    if not isinstance(signals, dict):
        signals = {}

    # The model outputs are not independent in a strict statistical sense;
    # therefore this check uses conservative weights and a very small cap.
    base_weights = {
        "markov": 0.13, "bayes": 0.13, "pattern": 0.12,
        "memory": 0.10, "stability": 0.10, "low_series": 0.08,
        "magnitude": 0.08, "context": 0.10, "ai_score": 0.16,
    }
    values, used_weights = [], []
    memory_conf = clamp(safe_float((memory_result or {}).get("confidence"), 0.0), 0.0, 1.0)
    for name, weight in base_weights.items():
        if name not in signals:
            continue
        value = _v6_norm_signal(signals.get(name), 0.5)
        if name == "memory":
            weight *= memory_conf
        if weight <= 0:
            continue
        values.append((name, value, weight))

    # A historical success rate is admitted only with enough completed,
    # sequence-similar observations; tiny cohorts cannot steer the decision.
    historical = result.get("historical_comparison") or {}
    cohort = historical.get("all_similar") if isinstance(historical, dict) else {}
    historical_n = int(safe_float((cohort or {}).get("samples"), 0))
    historical_rate = (cohort or {}).get("weighted_gain_rate")
    if historical_n >= 8 and historical_rate is not None:
        hist_value = clamp(safe_float(historical_rate, 0.5), 0.0, 1.0)
        hist_weight = min(0.10, 0.025 + historical_n / 400.0)
        values.append(("historical", hist_value, hist_weight))

    total_weight = sum(item[2] for item in values)
    if len(values) < 3 or total_weight <= 0:
        audit = {"version": 1, "status": "insufficient_independent_signals",
                 "signals_used": len(values), "historical_samples": historical_n,
                 "score_before": round(current, 6), "score_after": round(current, 6),
                 "adjustment": 0.0, "dispersion": None, "consensus": None}
        result["final_decision_audit"] = audit
        return result

    consensus = sum(value * weight for _, value, weight in values) / total_weight
    dispersion = math.sqrt(sum(weight * (value - consensus) ** 2
                               for _, value, weight in values) / total_weight)
    agreement = clamp(1.0 - dispersion * 2.5, 0.0, 1.0)
    evidence = clamp(min(1.0, len(values) / 7.0) * (0.55 + 0.45 * memory_conf), 0.0, 1.0)

    # Consensus can nudge the score only when the engines agree and evidence
    # is adequate. Otherwise the gate slightly shrinks overconfident extremes.
    if agreement >= 0.62 and evidence >= 0.45:
        strength = min(0.08, 0.025 + 0.055 * agreement * evidence)
        adjustment = clamp((consensus - current) * strength, -0.025, 0.025)
        status = "consensus_nudge"
    elif dispersion >= 0.18:
        shrink = min(0.015, (dispersion - 0.18) * 0.06)
        adjustment = -math.copysign(shrink, current - 0.5) if current != 0.5 else 0.0
        status = "disagreement_shrink"
    else:
        adjustment = 0.0
        status = "neutral_check"

    adjusted = clamp(current + adjustment, 0.0, 1.0)
    result["score"] = round(adjusted, 4)
    result["pertinence"] = round(adjusted * 100.0, 2)
    # Do not inflate confidence because of consensus; only reduce it when
    # material disagreement is detected. This keeps confidence conservative.
    old_confidence = clamp(safe_float(result.get("confidence"), current), 0.0, 1.0)
    confidence_penalty = min(0.06, max(0.0, dispersion - 0.12) * 0.15)
    result["confidence"] = round(clamp(old_confidence - confidence_penalty, 0.0, 1.0), 4)
    result["final_decision_audit"] = {
        "version": 1, "status": status, "signals_used": len(values),
        "signal_names": [name for name, _, _ in values],
        "historical_samples": historical_n,
        "consensus": round(consensus, 6), "agreement": round(agreement, 6),
        "evidence": round(evidence, 6), "dispersion": round(dispersion, 6),
        "score_before": round(current, 6), "score_after": round(adjusted, 6),
        "adjustment": round(adjustment, 6),
        "confidence_penalty": round(confidence_penalty, 6),
    }
    return result

# ══════════════════════════════════════════════════════════════════════
# V6 — REAL ADVANCED ML ENSEMBLE
# ══════════════════════════════════════════════════════════════════════
_ADVANCED_ML_CACHE = {"dataset_n": -1, "models": [], "weights": {}, "metrics": {}, "cluster": None}


def _advanced_ml_training_rows():
    rows=[]
    for item in (training_dataset or []):
        if not isinstance(item, dict):
            continue
        f=item.get("features")
        y=item.get("label")
        if not isinstance(f,(list,tuple)) or len(f)!=12:
            continue
        try:
            yy=float(y)
            if yy not in (0.0,1.0):
                continue
            ff=[clamp(float(x),0.0,1.0) for x in f]
            if all(math.isfinite(x) for x in ff):
                rows.append((ff,yy))
        except Exception:
            continue
    return rows[-5000:]


def _advanced_ml_fit():
    global _ADVANCED_ML_CACHE
    if not SKLEARN_AVAILABLE:
        return _ADVANCED_ML_CACHE
    rows=_advanced_ml_training_rows()
    n=len(rows)
    if n < 40 or len({int(y) for _,y in rows}) < 2:
        return _ADVANCED_ML_CACHE
    if _ADVANCED_ML_CACHE.get("dataset_n") == n and _ADVANCED_ML_CACHE.get("models"):
        return _ADVANCED_ML_CACHE
    X=np.asarray([x for x,_ in rows],dtype=float); y=np.asarray([y for _,y in rows],dtype=int)
    split=max(25,int(n*0.80)); split=min(split,n-10)
    Xtr,Xv=X[:split],X[split:]; ytr,yv=y[:split],y[split:]
    candidates=[]
    try: candidates.append(("random_forest",RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=2,random_state=42,class_weight="balanced_subsample",n_jobs=-1)))
    except Exception: pass
    try: candidates.append(("gradient_boosting",HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=0.05,l2_regularization=0.5,random_state=42)))
    except Exception: pass
    try: candidates.append(("mlp",make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32,16),activation="relu",solver="adam",alpha=0.001,learning_rate_init=0.003,max_iter=500,early_stopping=True,random_state=42))))
    except Exception: pass
    if XGBOOST_AVAILABLE:
        try: candidates.append(("xgboost",XGBClassifier(n_estimators=220,max_depth=5,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,eval_metric="logloss",random_state=42,n_jobs=2)))
        except Exception: pass
    if LIGHTGBM_AVAILABLE:
        try: candidates.append(("lightgbm",LGBMClassifier(n_estimators=220,num_leaves=15,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,verbosity=-1,random_state=42)))
        except Exception: pass
    trained=[]; quality={}
    base=float(np.mean(ytr))
    for name,model in candidates:
        try:
            model.fit(Xtr,ytr); p=np.asarray(model.predict_proba(Xv)[:,1],dtype=float)
            brier=float(np.mean((p-yv)**2)); ll=float(-np.mean(yv*np.log(np.clip(p,1e-7,1-1e-7))+(1-yv)*np.log(np.clip(1-p,1e-7,1-1e-7))))
            quality[name]={"brier":brier,"logloss":ll,"n_validation":len(yv)}
            trained.append((name,model))
        except Exception as e:
            logging.warning("[V6 ML] %s training skipped: %s",name,e)
    # KMeans is a regime detector; its cluster-level gain rates become a separate signal.
    cluster_model=None; cluster_rates={}
    try:
        k=min(5,max(2,len(Xtr)//20)); cluster_model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xtr)
        labels=cluster_model.labels_
        for c in range(k):
            ys=ytr[labels==c]; cluster_rates[c]=float(np.mean(ys)) if len(ys) else base
    except Exception as e:
        logging.warning("[V6 ML] KMeans skipped: %s",e)
    if trained:
        raw={name:1.0/(quality[name]["brier"]+0.01) * 1.0/(quality[name]["logloss"]+0.05) for name,_ in trained}
        total=sum(raw.values()) or 1.0; weights={k:v/total for k,v in raw.items()}
        _ADVANCED_ML_CACHE={"dataset_n":n,"models":trained,"weights":weights,"metrics":quality,"cluster":(cluster_model,cluster_rates),"base":base}
        logging.info("[V6 ML] trained=%s n=%d weights=%s",[x[0] for x in trained],n,{k:round(v,3) for k,v in weights.items()})
    return _ADVANCED_ML_CACHE


def v6_advanced_ml_predict(features):
    cache=_advanced_ml_fit()
    models=cache.get("models") or []
    if not models:
        return {"available":False,"score":0.5,"agreement":0.0,"models":{}}
    X=np.asarray(features,dtype=float).reshape(1,-1); outputs={}; vals=[]
    for name,model in models:
        try:
            p=float(model.predict_proba(X)[0,1]); outputs[name]=p; vals.append(p)
        except Exception:
            continue
    if not vals:
        return {"available":False,"score":0.5,"agreement":0.0,"models":{}}
    weights=cache.get("weights") or {}; z=sum(weights.get(k,0) for k in outputs) or 1.0
    score=sum(outputs[k]*weights.get(k,0) for k in outputs)/z
    agreement=clamp(1.0-2.5*float(np.std(vals)),0.0,1.0)
    cluster_signal=None
    try:
        km,rates=cache.get("cluster") or (None,{})
        if km is not None:
            c=int(km.predict(X)[0]); cluster_signal=float(rates.get(c,cache.get("base",0.5)))
            score=0.88*score+0.12*cluster_signal
    except Exception: pass
    return {"available":True,"score":round(clamp(score,0,1),6),"agreement":round(agreement,6),"models":{k:round(v,6) for k,v in outputs.items()},"metrics":cache.get("metrics",{})}


# PREDICTION ENGINE — CORRECTED
# ══════════════════════════════════════════════════════════════════════

def run_prediction(
    cotes,
    mode,
    ai_score,
    pattern_result
) -> Optional[dict]:

    last = (
        cotes[-ANALYSE_LAST:]
        if len(cotes) >= ANALYSE_LAST
        else list(cotes)
    )

    if not last:
        return None

    m, s = mean_std(last)
    low = consecutive_low(last)
    stab = stability_score(last)
    markov = weighted_markov_next(last)
    bayes = bayes_probability(last)
    ent = entropy_score(last)

    pattern_result = (
        pattern_result
        if isinstance(pattern_result, dict)
        else {}
    )

    pat_bonus = _safe_float(
        pattern_result.get(
            "net",
            0.0
        )
    )

    # Real supervised ensemble (RF + Gradient Boosting + MLP, with optional
    # XGBoost/LightGBM) is trained only on chronological feedback samples.
    ml_features = build_prediction_features(
        consecutive_low_value=low, volatility_value=volatility(last),
        std_value=s, mean_value=m, predicted_score=0.0, predicted_target=0.0,
        predicted_hour=datetime.datetime.now().hour, predicted_minute=datetime.datetime.now().minute,
        peak_interval=compute_peak_interval(last) or 0.0, sequence_length=len(last),
        pattern_gain=pattern_result.get("raw_gain",0.0), pattern_loss=pattern_result.get("raw_loss",0.0)
    )
    advanced_ml = v6_advanced_ml_predict(ml_features)
    advanced_score = safe_float(advanced_ml.get("score"), score if "score" in locals() else 0.5)

    # ==============================================================
    # DIRECTION — séparée du TIMING
    # ==============================================================

    min_t, max_t = TARGET_RANGES.get(
        mode,
        (2.0, 5.0)
    )

    score = clamp(
        _safe_float(
            ai_score,
            0.0
        ),
        0.0,
        1.0
    )
    if advanced_ml.get("available"):
        score = clamp(0.45*score + 0.55*advanced_score, 0.0, 1.0)

    # Evidence from explicitly learned hot/weak/fall/rise/transition families.
    # It is neutral until enough directional examples exist; timing labels never
    # contribute to this score.
    sequence_memory_signal = get_sequence_feature_direction_score(last, mode)
    if sequence_memory_signal.get("confidence", 0.0) > 0:
        blend = min(0.12, 0.12 * sequence_memory_signal["confidence"])
        score = clamp((1.0 - blend) * score + blend * sequence_memory_signal["score"], 0.0, 1.0)

    if mode == "2X-10X":

        base = (
            m
            + 0.5 * s
            + 0.3 * score * m
        )

    elif mode == "grosse":

        base = (
            m
            + (
                2.0
                + 4.0 * score
            )
            * max(s, 0.2)
            + 1.5 * low
            + 2.5 * pat_bonus
        )

    else:

        base = (
            m
            + (
                10.0
                + 20.0 * score
            )
            * max(s, 0.2)
            + 4.0 * low
            + 8.0 * pat_bonus
        )

    target = round(
        clamp(
            base,
            min_t,
            max_t
        ),
        2
    )

    target_bias = v6_target_bias_correction(mode)
    if target_bias.get("samples", 0) >= 10:
        target = round(clamp(target * safe_float(target_bias.get("factor"), 1.0), min_t, max_t), 2)

    # Placeholder directionnel.
    # Le calcul final de confiance peut être effectué
    # par le module supérieur après fusion mémoire/ensemble.
    assurance = round(
        target * 0.50,
        2
    )

    fiabilite = round(
        target * 0.65,
        2
    )

    pertinence = round(
        score * 100.0,
        1
    )

    # ==============================================================
    # CONTEXTE TEMPOREL
    #
    # IMPORTANT :
    # aucune décision directe "low => 1 min"
    # ou "stable => 3 min".
    #
    # Ces informations entrent dans le modèle temporel.
    # ==============================================================

    context = detect_market_context(
        last
    )

    pattern_key = build_pattern_key(
        last
    )

    timing_context_key = _timing_context_key(
        mode,
        last,
        context=context,
        pattern_key=pattern_key
    )

    # ==============================================================
    # 1. PRÉDICTION DU MODÈLE DE DÉLAI
    # ==============================================================

    delay_predictions = (
        get_learned_delay_predictions(
            last,
            mode,
            timing_context_key
        )
    )

    # ==============================================================
    # 2. MÉMOIRE DE DÉLAIS OBSERVÉS
    # ==============================================================

    observed_scores = (
        get_observed_timing_scores(
            mode,
            last,
            timing_context_key
        )
    )

    # ==============================================================
    # 3. COMPÉTITION ADAPTATIVE
    # ==============================================================

    pattern_timing_distribution = (
        get_pattern_timing_distribution(
            last,
            mode
        )
    )

    pattern_timing_scores = {}

    for delay in SUPPORTED_DELAYS:
        info = get_pattern_timing_score(
            last,
            mode,
            delay
        )
        base_pattern_score = info.get("score", 0.50)
        feature_timing = get_sequence_feature_timing_score(last, mode, delay)
        effective_feature_support = _safe_float(feature_timing.get("samples", 0), 0.0) + 0.75 * min(6, int(_safe_float(feature_timing.get("suggestion_samples", 0), 0)))
        if effective_feature_support >= SEQUENCE_FEATURE_MIN_SAMPLES:
            # Measured outcomes carry full weight; repeated +/-1 hints count
            # below full weight and can never exceed a 25% blend.
            blend = min(0.25, 0.25 * effective_feature_support / 6.0)
            base_pattern_score = (1.0 - blend) * base_pattern_score + blend * feature_timing["score"]
        pattern_timing_scores[delay] = base_pattern_score

    delay_scores = compute_delay_competition(
        delay_predictions,
        pattern_scores=None,
        recent_context={
            "timing_context_key": timing_context_key,
            "mode": mode,
            "pattern_sequence": list(last),
        },
        observed_scores=observed_scores,
        pattern_timing_scores=pattern_timing_scores
    )

    minute_offset, live_delay_probabilities = live_timing_decision(
        delay_scores,
        context_key=timing_context_key
    )

    # ==============================================================
    # 4. DISTRIBUTION DES DÉLAIS
    # ==============================================================

    raw_delay_scores = {
        delay:
            max(
                0.0001,
                _safe_float(
                    data.get(
                        "final_score",
                        0.0
                    )
                )
            )
        for delay, data
        in delay_scores.items()
    }

    total_delay_score = sum(
        raw_delay_scores.values()
    )

    if total_delay_score <= 0:

        delay_probabilities = {
            delay:
                1.0 / max(
                    len(SUPPORTED_DELAYS),
                    1
                )
            for delay in SUPPORTED_DELAYS
        }

    else:

        delay_probabilities = dict(live_delay_probabilities)

    # ==============================================================
    # 5. CONFIANCE TEMPORELLE
    # ==============================================================

    sorted_probs = sorted(
        delay_probabilities.values(),
        reverse=True
    )

    best_prob = (
        sorted_probs[0]
        if sorted_probs
        else 0.0
    )

    second_prob = (
        sorted_probs[1]
        if len(sorted_probs) > 1
        else 0.0
    )

    # Plus la première fenêtre se détache,
    # plus le délai est déterminé.
    timing_confidence = clamp(
        0.50
        + (
            best_prob
            - second_prob
        ),
        0.0,
        1.0
    )

    # ==============================================================
    # 6. SECONDES À L'INTÉRIEUR DE LA FENÊTRE
    # ==============================================================

    # IMPORTANT :
    # Les secondes ne doivent plus être calculées avec
    # timing_urgency dépendant du score directionnel.
    #
    # On utilise la position moyenne des expériences observées
    # lorsque disponible.

    weighted_observed_delay = None

    matching_observations = []

    for exp in experience_memory:

        similarity = (
            _timing_experience_similarity(
                build_current_context(
                    last,
                    mode,
                    0.0
                ),
                exp,
                mode
            )
        )

        if similarity <= 0:
            continue

        # IMPORTANT: only verified/observed timing is eligible here.
        # Never fall back to the predicted delay.
        observed = exp.get("actual_delay_sec")
        if observed is None:
            observed = exp.get("actual_delay")
        if observed is None:
            observed = exp.get("delay_seconds")

        observed = _safe_float(
            observed,
            -1
        )

        if math.isfinite(observed) and 0 < observed <= 1800:
            matching_observations.append(
                (
                    similarity,
                    observed
                )
            )

    if matching_observations:

        numerator = sum(
            similarity * observed
            for similarity, observed
            in matching_observations[:MAX_SIMILAR_EXPERIENCES]
        )

        denominator = sum(
            similarity
            for similarity, _
            in matching_observations[:MAX_SIMILAR_EXPERIENCES]
        )

        if denominator > 0:

            weighted_observed_delay = (
                numerator
                / denominator
            )

    if weighted_observed_delay is not None:

        # On conserve le centre appris dans une zone souple.
        target_delay_seconds = (
            weighted_observed_delay
        )

    else:

        # Sans expérience historique, on prend le centre
        # de la fenêtre gagnante.
        target_delay_seconds = (
            _delay_center(
                minute_offset
            )
        )

    # ==============================================================
    # 7. CONTRAINTE À LA FENÊTRE GAGNANTE
    # ==============================================================

    window_start_seconds = (
        minute_offset * 60
    )

    window_end_seconds = (
        (minute_offset + 1) * 60
    )

    # Les fenêtres restent cohérentes :
    # le délai choisi reste bien dans la minute choisie.
    delay_sec = int(
        clamp(
            target_delay_seconds,
            window_start_seconds,
            window_end_seconds - 1
        )
    )

    now = datetime.datetime.now()

    optimal_time = (
        now
        + datetime.timedelta(
            seconds=delay_sec
        )
    )

    win_start = (
        now
        + datetime.timedelta(
            minutes=minute_offset
        )
    )

    win_end = (
        win_start
        + datetime.timedelta(
            seconds=60
        )
    )

    # ==============================================================
    # STRATÉGIE DIRECTIONNELLE
    # ==============================================================

    if score >= 0.70:

        strategy = (
            "🔥 Signal directionnel fort"
        )

    elif score >= 0.50:

        strategy = (
            "⚡ Signal directionnel moyen"
        )

    else:

        strategy = (
            "⚠️ Signal directionnel faible"
        )

    total_low = sum(
        1
        for x in last
        if x < 2.0
    )

    # ==============================================================
    # v6 — ENSEMBLE / NOVELTY / ABSTENTION / TIMING INTERVAL
    # ==============================================================

    direction_probability, direction_confidence, direction_components, ensemble_agreement = (
        v6_ensemble_direction(
            last, ai_score, pattern_result, bayes, markov, stab, ent
        )
    )

    novelty_score, historical_support, historical_similarity = v6_experience_novelty(
        last, mode
    )

    timing_interval = v6_timing_interval(
        {str(k): v for k, v in delay_probabilities.items()},
        coverage=V6_TIMING_INTERVAL_COVERAGE
    )

    timing_entropy = v6_timing_entropy(delay_probabilities)
    timing_confidence = clamp(
        0.55 * timing_confidence
        + 0.25 * (1.0 - timing_entropy)
        + 0.20 * (1.0 if DELAY_MODEL_STATUS == "trained" else 0.0),
        0.0,
        1.0
    )

    decision, decision_confidence, decision_reasons = v6_decision_gate(
        direction_probability,
        direction_confidence,
        timing_confidence,
        ensemble_agreement,
        novelty_score,
        DELAY_MODEL_STATUS
    )

    prediction_id = str(uuid.uuid4())
    magnitude_model = v6_magnitude_distribution(last, mode)
    uncertainty = {"direction":round(1-direction_confidence,4),"timing":round(1-timing_confidence,4),"magnitude":round(1-magnitude_model["confidence"],4),"novelty":round(novelty_score,4),"overall":round(clamp(0.35*(1-direction_confidence)+0.35*(1-timing_confidence)+0.20*(1-magnitude_model["confidence"])+0.10*novelty_score,0,1),4)}

    # ==============================================================
    # RÉSULTAT
    # ==============================================================

    # Advanced-model diagnostics are retained for later validation/reweighting.
    result = {
        "advanced_ml_available": bool(advanced_ml.get("available")),
        "advanced_ml_score": round(safe_float(advanced_ml.get("score"),0.5),6),
        "advanced_ml_agreement": round(safe_float(advanced_ml.get("agreement"),0.0),6),
        "advanced_ml_models": dict(advanced_ml.get("models",{}) or {}),

        "target": target,
        "assurance": assurance,
        "fiabilite": fiabilite,
        "pertinence": pertinence,

        "moyenne": round(
            m,
            2
        ),

        "ecart": round(
            s,
            3
        ),

        "serie_creuse": total_low,

        "score": round(
            score,
            3
        ),

        # v6 — separate directional probability from the legacy score.
        "prediction_id": prediction_id,
        "direction_probability": round(direction_probability, 4),
        "direction_confidence": round(direction_confidence, 4),
        "direction_components": {k: round(v, 4) for k, v in direction_components.items()},
        "magnitude_distribution": magnitude_model["probabilities"],
        "magnitude_confidence": magnitude_model["confidence"],
        "magnitude_model_status": magnitude_model["status"],
        "magnitude_samples": magnitude_model["samples"],
        "uncertainty": uncertainty,
        "target_bias_correction": target_bias,
        "model_agreement": round(ensemble_agreement, 4),
        "novelty_score": round(novelty_score, 4),
        "historical_support": historical_support,
        "historical_similarity": round(historical_similarity, 4),
        "decision": decision,
        "decision_confidence": round(decision_confidence, 4),
        "decision_reasons": decision_reasons,

        "stability": round(
            stab,
            3
        ),

        "pattern": pattern_result,

        "markov": round(
            markov,
            3
        ),

        "bayes": round(
            bayes,
            3
        ),

        "entropy": round(
            ent,
            3
        ),

        # ----------------------------------------------------------
        # TIMING
        # ----------------------------------------------------------

        "optimal_time": (
            optimal_time.strftime(
                "%H:%M:%S"
            )
        ),

        "fenetre": (
            win_start.strftime(
                "%H:%M:%S"
            ),
            win_end.strftime(
                "%H:%M:%S"
            )
        ),

        "delay_sec": delay_sec,
        "delay_min": minute_offset,

        "delay_scores": delay_scores,

        "delay_probabilities": {
            str(delay):
                round(
                    probability,
                    4
                )
            for delay, probability
            in delay_probabilities.items()
        },

        "timing_confidence": round(
            timing_confidence,
            4
        ),

        "timing_interval": _v6_timing_interval_from_residual(delay_sec),
        "timing_entropy": round(timing_entropy, 4),
        "timing_model_status": DELAY_MODEL_STATUS,
        "timing_window_center_seconds": delay_sec,

        "timing_context_key":
            timing_context_key,

        "timing_evidence": {

            "model_predictions":
                delay_predictions,

            "observed_scores":
                observed_scores,

            "context_key":
                timing_context_key,

            "historical_observations":
                len(
                    matching_observations
                ),

            "weighted_observed_delay":
                (
                    round(
                        weighted_observed_delay,
                        2
                    )
                    if weighted_observed_delay
                    is not None
                    else None
                ),

            "pattern_timing_distribution": {
                str(delay): round(
                    score,
                    4
                )
                for delay, score
                in pattern_timing_distribution.items()
            },

            "source":
                "learned_delay_model"
                "+verified_timing_memory"
                "+historical_timing"
                "+hour_memory"
                "+pattern_delay_memory"
        },

        # ----------------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------------

        "strategy": strategy,

        "pattern_sequence":
            list(last)
    }

    v6_record_prediction(result, mode, last)

    return result



# ══════════════════════════════════════════════════════════════════════
# ULTRA FILTER
# ══════════════════════════════════════════════════════════════════════

def filtre_ultra(cotes):

    if len(cotes) < 15:

        return (
            False,
            "⚠️ Trop peu de cotes pour ultra."
        )

    recent = list(
        cotes[-20:]
    )

    m = float(
        np.mean(recent)
    )

    s = float(
        np.std(recent)
    )

    if any(
        x >= 10
        for x in cotes[-15:]
    ):

        return (
            False,
            "⚠️ Gros pic déjà présent."
        )

    if m > 2.0:

        return (
            False,
            "⚠️ Moyenne trop haute."
        )

    if s > 1.5:

        return (
            False,
            "⚠️ Série instable."
        )

    return (
        True,
        "✅ Conditions OK."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 4/4 — MEMORY / OCR / FEEDBACK / TELEGRAM / FLASK / MAIN
# Rocket Queen V7 — compatibilité des anciens helpers
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# DEPENDENCY GUARDS / HELPERS
# ══════════════════════════════════════════════════════════════

if "safe_float" not in globals():

    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default


if "mean_std" not in globals():

    def mean_std(values):
        vals = []
        for v in values or []:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass

        if not vals:
            return 0.0, 0.0

        mean = sum(vals) / len(vals)

        if len(vals) <= 1:
            return mean, 0.0

        variance = sum((x - mean) ** 2 for x in vals) / len(vals)

        try:
            std = math.sqrt(max(variance, 0.0))
        except Exception:
            std = 0.0

        return mean, std


if "volatility" not in globals():

    def volatility(values):
        _, std = mean_std(values)
        mean, _ = mean_std(values)

        if mean <= 0:
            return std

        return std / mean


if "consecutive_low" not in globals():

    def consecutive_low(values, threshold=1.99):
        count = 0

        for value in reversed(values or []):
            try:
                if float(value) <= threshold:
                    count += 1
                else:
                    break
            except (TypeError, ValueError):
                break

        return count


if "weighted_markov_next" not in globals():

    def weighted_markov_next(values):
        vals = []

        for v in values or []:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass

        if len(vals) < 2:
            return 0.5

        low = 0
        total = 0

        for a, b in zip(vals[:-1], vals[1:]):
            total += 1

            if b <= 1.99:
                low += 1

        if total == 0:
            return 0.5

        # Retourne une probabilité "issue favorable / non-faible"
        return max(0.0, min(1.0, 1.0 - (low / total)))


if "compute_peak_interval" not in globals():

    def compute_peak_interval(values):
        vals = list(values or [])

        if len(vals) < 2:
            return 0.0

        peak_indices = []

        for i, value in enumerate(vals):
            try:
                if float(value) >= 10.0:
                    peak_indices.append(i)
            except (TypeError, ValueError):
                pass

        if len(peak_indices) < 2:
            return 0.0

        intervals = [
            b - a
            for a, b in zip(peak_indices[:-1], peak_indices[1:])
        ]

        return (
            sum(intervals) / len(intervals)
            if intervals else 0.0
        )


if "html_mod" not in globals():
    import html as html_mod


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MEMORY UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_pattern_weights():
    """
    Maintient SEQ_WEIGHTS dans une plage saine puis normalise leur somme à 1.
    """
    try:
        valid_lengths = [L for L in (5, 10, 15) if L in SEQ_WEIGHTS]

        if not valid_lengths:
            return

        for L in valid_lengths:
            value = safe_float(SEQ_WEIGHTS.get(L), 0.0)

            # Bornes de sécurité
            value = max(0.10, min(0.70, value))
            SEQ_WEIGHTS[L] = value

        total = sum(SEQ_WEIGHTS[L] for L in valid_lengths)

        if total <= 0:
            equal = 1.0 / len(valid_lengths)
            for L in valid_lengths:
                SEQ_WEIGHTS[L] = equal
        else:
            for L in valid_lengths:
                SEQ_WEIGHTS[L] /= total

    except Exception as e:
        logging.warning(f"[PATTERN WEIGHTS] {e}")


def _pattern_exists_exact(seq, collection):
    """
    Recherche exacte d'une séquence dans une collection.
    """
    try:
        seq_list = list(seq)
        return any(list(item) == seq_list for item in collection)
    except Exception:
        return False


def _store_pattern_once(pattern_seq, target_key):
    """
    Ajoute un pattern à la mémoire sans créer de doublon.
    """
    if target_key not in ("gains", "pertes"):
        return False

    stored_any = False

    for L in (5, 10, 15):
        if len(pattern_seq) < L:
            continue

        seq = list(pattern_seq[-L:])
        collection = pattern_memory.setdefault(target_key, {}).setdefault(L, [])

        if not _pattern_exists_exact(seq, collection):
            collection.append(seq)

            if len(collection) > MAX_PATTERNS:
                del collection[0:len(collection) - MAX_PATTERNS]

            stored_any = True

    return stored_any


# ══════════════════════════════════════════════════════════════════════
# PATTERN → DÉLAI → RÉSULTAT — MÉMOIRE TEMPORELLE v6.0
# ══════════════════════════════════════════════════════════════════════

PATTERN_DELAY_MIN_SAMPLES = 3
PATTERN_DELAY_MAX_ENTRIES = 5000
PATTERN_DELAY_MEMORY_KEY = "timing_v6_adaptive"


def _pattern_timing_key(pattern_seq, mode):
    """Transforme les cinq dernières cotes en famille comportementale."""

    values = list(pattern_seq or [])[-5:]
    if not values:
        return None

    classes = []

    for value in values:
        value = _safe_float(value, 0.0)

        if value < 1.30:
            classes.append("L")
        elif value < 2.00:
            classes.append("M")
        elif value < 5.00:
            classes.append("H")
        else:
            classes.append("X")

    while len(classes) < 5:
        classes.insert(0, "U")

    return (
        f"{mode or 'unknown'}|"
        + "_".join(classes)
    )


def _pattern_timing_entry(pattern_key, delay):
    """Retourne la case mémoire d'une famille et d'une fenêtre."""

    global pattern_stats

    if not pattern_key:
        return None

    try:
        delay = int(delay)
    except Exception:
        return None

    if delay not in SUPPORTED_DELAYS:
        return None

    if not isinstance(pattern_stats, dict):
        pattern_stats = {}

    root = pattern_stats.setdefault(PATTERN_DELAY_MEMORY_KEY, {})
    if not isinstance(root, dict):
        root = {}
        pattern_stats[PATTERN_DELAY_MEMORY_KEY] = root
    pattern = root.setdefault(pattern_key, {})

    return pattern.setdefault(
        str(delay),
        {
            "total": 0,
            "timing_success": 0,
            "timing_loss": 0,
            "early": 0,
            "late": 0,
            "observed_delays": [],
            "recent_results": [],
            "confidence_sum": 0.0,
        }
    )


def update_pattern_timing_memory(
    pattern_seq,
    mode,
    predicted_delay,
    actual_delay,
    status,
    confidence=0.50,
    actual_delay_source=None
):
    """Apprend pattern → délai → justesse temporelle, sans étiquette cible."""
    try:
        predicted_delay = int(predicted_delay)
    except Exception:
        return
    if predicted_delay not in SUPPORTED_DELAYS:
        return

    status = str(status or "").lower().strip()
    if status not in ("gain", "perte", "loss", "early", "late"):
        return
    actual_delay = _safe_float(actual_delay, -1.0)
    measured = actual_delay_source == "observed_datetime" and 0 < actual_delay <= 1800
    if status in ("gain", "perte", "loss") and not measured:
        return
    if status in ("early", "late") and actual_delay_source == "observed_datetime" and not measured:
        return

    values = list(pattern_seq or [])[-5:]
    if not values:
        return

    keys = []
    exact_key = _pattern_timing_key(values, mode)
    if exact_key:
        keys.append(exact_key)
    classes = []
    for value in values:
        value = _safe_float(value, 0.0)
        classes.append("L" if value < 1.30 else "M" if value < 2.00 else "H" if value < 5.00 else "X")
    for length in (4, 3):
        if len(classes) >= length:
            sub_key = f"{mode or 'unknown'}|" + "_".join(classes[-length:])
            if sub_key not in keys:
                keys.append(sub_key)

    # Explicit early/late feedback means the selected delay was wrong.
    timing_correct = status not in ("early", "late") and _delay_bucket(actual_delay) == predicted_delay and abs(actual_delay - _delay_center(predicted_delay)) <= 30.0
    confidence = clamp(_safe_float(confidence, 0.50), 0.0, 1.0)
    timing_root = pattern_stats.setdefault(PATTERN_DELAY_MEMORY_KEY, {})

    for pattern_key in keys:
        entry = _pattern_timing_entry(pattern_key, predicted_delay)
        if entry is None:
            continue
        entry["total"] = int(_safe_float(entry.get("total", 0), 0)) + 1
        counter = "timing_success" if timing_correct else "timing_loss"
        entry[counter] = int(_safe_float(entry.get(counter, 0), 0)) + 1
        if status == "early":
            entry["early"] = int(_safe_float(entry.get("early", 0), 0)) + 1
        elif status == "late":
            entry["late"] = int(_safe_float(entry.get("late", 0), 0)) + 1
        if measured:
            observed = entry.setdefault("observed_delays", [])
            observed.append(actual_delay)
            entry["observed_delays"] = observed[-50:]
        entry["confidence_sum"] = _safe_float(entry.get("confidence_sum", 0.0), 0.0) + confidence
        recent = entry.setdefault("recent_results", [])
        recent.append({
            "status": "timing_success" if timing_correct else "timing_loss",
            "directional_status": status,
            "predicted_delay": predicted_delay,
            "actual_delay": actual_delay if measured else None,
            "actual_delay_source": actual_delay_source,
            "confidence": confidence,
            "time": datetime.datetime.now().isoformat(),
        })
        entry["recent_results"] = recent[-30:]

    while len(timing_root) > PATTERN_DELAY_MAX_ENTRIES:
        timing_root.pop(next(iter(timing_root)))


def get_pattern_timing_score(pattern_seq, mode, delay):
    """Retourne le score et la fiabilité d'une fenêtre pour un pattern."""

    pattern_key = _pattern_timing_key(pattern_seq, mode)
    if not pattern_key:
        return {
            "score": 0.50,
            "samples": 0,
            "confidence": 0.0,
            "reliability": 0.0,
        }

    try:
        delay = int(delay)
    except Exception:
        delay = 0

    timing_root = (
        pattern_stats.get(PATTERN_DELAY_MEMORY_KEY, {})
        if isinstance(pattern_stats, dict)
        else {}
    )
    if not isinstance(timing_root, dict):
        timing_root = {}

    entry = (
        timing_root
        .get(pattern_key, {})
        .get(str(delay))
    )

    # Exact pattern evidence remains primary.  If it is weak, also inspect
    # shorter suffix patterns (sub-patterns) without allowing one occurrence
    # to dominate the decision.
    sub_entries = []
    try:
        values = list(pattern_seq or [])[-5:]
        classes = []
        for value in values:
            value = _safe_float(value, 0.0)
            if value < 1.30:
                classes.append("L")
            elif value < 2.00:
                classes.append("M")
            elif value < 5.00:
                classes.append("H")
            else:
                classes.append("X")
        for length, weight in ((4, 0.75), (3, 0.55)):
            if len(classes) < length:
                continue
            sub_key = f"{mode or 'unknown'}|" + "_".join(classes[-length:])
            sub_entry = timing_root.get(sub_key, {}).get(str(delay))
            if isinstance(sub_entry, dict):
                sub_total = int(_safe_float(sub_entry.get("total", 0), 0))
                if sub_total >= PATTERN_DELAY_MIN_SAMPLES:
                    sub_entries.append((sub_entry, weight))
    except Exception:
        sub_entries = []

    if not isinstance(entry, dict):
        entry = {}

    exact_total = int(_safe_float(entry.get("total", 0), 0))
    if exact_total < PATTERN_DELAY_MIN_SAMPLES and not sub_entries:
        return {
            "score": 0.50,
            "samples": exact_total,
            "confidence": 0.0,
            "reliability": 0.0,
        }

    total = int(_safe_float(entry.get("total", 0), 0))
    success = int(_safe_float(entry.get("timing_success", 0), 0))
    loss = int(_safe_float(entry.get("timing_loss", 0), 0))

    # Blend verified suffix/sub-pattern evidence only when it has at least
    # the same minimum of 3 observations.  Exact evidence stays dominant.
    sub_weight_total = 0.0
    sub_success = 0.0
    sub_loss = 0.0
    for sub_entry, weight in sub_entries:
        st = int(_safe_float(sub_entry.get("total", 0), 0))
        ss = int(_safe_float(sub_entry.get("timing_success", 0), 0))
        sl = int(_safe_float(sub_entry.get("timing_loss", 0), 0))
        directional = ss + sl
        if st >= PATTERN_DELAY_MIN_SAMPLES and directional > 0:
            sub_weight_total += weight * min(1.0, st / 12.0)
            sub_success += weight * ss
            sub_loss += weight * sl

    if sub_weight_total > 0:
        sub_rate = sub_success / max(sub_success + sub_loss, 1.0)
        exact_directional = success + loss
        exact_rate = (success + 1.0) / (exact_directional + 2.0)
        exact_strength = min(1.0, exact_directional / 6.0) if exact_total >= PATTERN_DELAY_MIN_SAMPLES else 0.0
        combined_rate = exact_rate * exact_strength + sub_rate * (1.0 - exact_strength)
    else:
        combined_rate = None
    directional_total = success + loss

    historical_rate = (
        (success + 1.0) / (directional_total + 2.0)
        if directional_total > 0
        else 0.50
    )

    recent_success = 0
    recent_directional = 0

    for item in entry.get("recent_results", [])[-20:]:
        if not isinstance(item, dict):
            continue
        item_status = item.get("status")
        if item_status not in ("timing_success", "timing_loss"):
            continue
        recent_directional += 1
        if item_status == "timing_success":
            recent_success += 1

    recent_rate = (
        recent_success / recent_directional
        if recent_directional > 0
        else historical_rate
    )

    score = (
        historical_rate * 0.60
        + recent_rate * 0.40
    )
    if combined_rate is not None:
        # Sub-patterns supplement, never replace, exact verified evidence.
        score = score * 0.70 + combined_rate * 0.30

    # Suffix families are derived from the same feedback rows; use the best
    # supported sample count, not their sum, to avoid double-counting evidence.
    sub_support = max(
        (int(_safe_float(sub_entry.get("total", 0), 0)) for sub_entry, _ in sub_entries),
        default=0,
    )
    effective_support = max(float(total), float(sub_support))
    sample_factor = min(
        1.0,
        effective_support / float(PATTERN_DELAY_MIN_SAMPLES * 4)
    )
    confidence_values = [
        _safe_float(entry.get("confidence_sum", 0.0), 0.0) / max(total, 1)
    ] if total else []
    confidence_values.extend(
        _safe_float(sub_entry.get("confidence_sum", 0.0), 0.0)
        / max(int(_safe_float(sub_entry.get("total", 0), 0)), 1)
        for sub_entry, _ in sub_entries
    )
    confidence_avg = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    reliability = sample_factor * (0.50 + 0.50 * confidence_avg)

    # Evidence from sub-patterns is valid only with >=3 observations. Without
    # that support, the score stays neutral; sparse exact patterns cannot win.
    if effective_support < PATTERN_DELAY_MIN_SAMPLES:
        score = 0.50
        reliability = 0.0

    return {
        "score": clamp(score, 0.0, 1.0),
        "samples": int(round(effective_support)),
        "confidence": clamp(confidence_avg, 0.0, 1.0),
        "reliability": clamp(reliability, 0.0, 1.0),
    }


def get_pattern_timing_distribution(pattern_seq, mode):
    """Construit une distribution normalisée des fenêtres temporelles."""

    weighted = {}
    total = 0.0

    for delay in SUPPORTED_DELAYS:
        info = get_pattern_timing_score(
            pattern_seq,
            mode,
            delay
        )
        effective = (
            0.50
            + (
                info.get("score", 0.50)
                - 0.50
            ) * info.get("reliability", 0.0)
        )
        weighted[delay] = max(0.001, effective)
        total += weighted[delay]

    return {
        delay: weighted[delay] / total
        for delay in SUPPORTED_DELAYS
    }


# ──────────────────────────────────────────────────────────────────────
# MÉMOIRE EXPLICITE DES CARACTÉRISTIQUES DE SÉQUENCE v1
# Direction et timing sont stockés séparément dans chaque famille.
SEQUENCE_FEATURE_MEMORY_KEY = "sequence_feature_memory_v1"
SEQUENCE_FEATURE_MIN_SAMPLES = 3


def extract_sequence_features(sequence):
    """Caractéristiques explicites et reproductibles de la séquence observée.

    Seuils opérationnels : faible <1.99x ; niveau élevé >=5x ; une chute/montée
    exige une variation d'au moins 0.50x. Ces labels décrivent les cotes reçues,
    ils ne constituent pas une garantie de prédiction du prochain résultat.
    """
    try:
        values = [_safe_float(v, 0.0) for v in list(sequence or [])]
        values = [v for v in values if math.isfinite(v) and v > 0][-15:]
    except Exception:
        values = []
    if not values:
        return {"values": [], "labels": [], "signature": "empty", "metrics": {}}

    tail = values[-5:]
    classes = ["L" if v < 1.30 else "M" if v < 1.99 else "H" if v < 5.0 else "X" for v in tail]
    labels = ["level:hot"] if any(v >= 5.0 for v in tail) else []
    if len(tail) >= 3 and all(v < 1.99 for v in tail[-3:]):
        labels.append("level:weak_streak")
    if sum(v < 1.99 for v in tail) >= max(3, (len(tail) + 1) // 2):
        labels.append("level:weak_majority")
    if any(v >= 5.0 for v in tail[-3:]):
        labels.append("level:recent_high")

    diffs = [tail[i] - tail[i-1] for i in range(1, len(tail))]
    if diffs:
        if diffs[-1] <= -0.50:
            labels.append("shape:fall")
        if diffs[-1] >= 0.50:
            labels.append("shape:rise")
        if len(diffs) >= 2 and sum(d < 0 for d in diffs[-2:]) == 2:
            labels.append("shape:descending_run")
        if len(diffs) >= 2 and sum(d > 0 for d in diffs[-2:]) == 2:
            labels.append("shape:ascending_run")
        if abs(diffs[-1]) < 0.50:
            labels.append("shape:flat_last_step")
    if len(tail) >= 2:
        prev, last = tail[-2], tail[-1]
        if prev >= 5.0 and last < 1.99:
            labels.append("transition:high_to_weak")
        if prev < 1.99 and last >= 5.0:
            labels.append("transition:weak_to_high")
        if prev >= 1.99 and last < 1.99:
            labels.append("transition:to_weak")
        if prev < 1.99 and last >= 1.99:
            labels.append("transition:out_of_weak")

    # Always retain the exact coarse-grained ordered signature as well.
    labels.append("sequence:" + "_".join(classes))
    labels = list(dict.fromkeys(labels))
    metrics = {
        "n": len(tail),
        "mean": sum(tail) / len(tail),
        "low_count": sum(v < 1.99 for v in tail),
        "high_count": sum(v >= 5.0 for v in tail),
        "last_delta": diffs[-1] if diffs else 0.0,
        "max_drop": min(diffs) if diffs else 0.0,
        "max_rise": max(diffs) if diffs else 0.0,
    }
    return {"values": tail, "labels": labels, "signature": "_".join(classes), "metrics": metrics}


def _sequence_feature_entry(root, key):
    return root.setdefault(key, {
        "direction": {"total": 0, "gain": 0, "perte": 0, "recent": []},
        "timing_by_delay": {},
    })


def update_sequence_feature_memory(sequence, mode, status, predicted_delay=None,
                                   actual_delay=None, actual_delay_source=None,
                                   timing_offset_seconds=None, confidence=0.5):
    """Enregistre les catégories séparément pour le moteur directionnel et le timing."""
    if not isinstance(pattern_stats, dict):
        return
    profile = extract_sequence_features(sequence)
    labels = profile.get("labels", [])
    if not labels:
        return
    root = pattern_stats.setdefault(SEQUENCE_FEATURE_MEMORY_KEY, {})
    if not isinstance(root, dict):
        root = {}
        pattern_stats[SEQUENCE_FEATURE_MEMORY_KEY] = root
    mode_key = str(mode or "unknown")
    status = str(status or "").lower().strip()
    confidence = clamp(_safe_float(confidence, 0.5), 0.0, 1.0)

    if status in ("gain", "perte", "loss"):
        outcome = "gain" if status == "gain" else "perte"
        for label in labels:
            entry = _sequence_feature_entry(root, f"{mode_key}|{label}")
            d = entry["direction"]
            d["total"] = int(d.get("total", 0)) + 1
            d[outcome] = int(d.get(outcome, 0)) + 1
            d.setdefault("recent", []).append({"outcome": outcome, "time": datetime.datetime.now().isoformat()})
            d["recent"] = d["recent"][-50:]

    # A timing label is only allowed to update timing buckets. A directional
    # result counts as timing evidence only when a real elapsed time was measured.
    observed = _safe_float(actual_delay, -1.0)
    measured = actual_delay_source == "observed_datetime" and 0 < observed <= 1800
    if status in ("early", "late") or (status in ("gain", "perte", "loss") and measured):
        try:
            selected = int(round(float(predicted_delay)))
        except Exception:
            selected = 0
        if selected not in SUPPORTED_DELAYS:
            selected = 0
        timing_correct = bool(measured and selected and _delay_bucket(observed) == selected and abs(observed - _delay_center(selected)) <= 30.0 and status not in ("early", "late"))
        for label in labels:
            entry = _sequence_feature_entry(root, f"{mode_key}|{label}")
            # Early/late only penalizes the selected window; it does not assert
            # a measured alternative window.
            if selected:
                bucket = entry["timing_by_delay"].setdefault(str(selected), {
                    "total": 0, "success": 0, "loss": 0, "early": 0, "late": 0,
                    "observed_delays": [], "recent": [],
                })
                bucket["total"] += 1
                if timing_correct:
                    bucket["success"] += 1
                else:
                    bucket["loss"] += 1
                if status == "early": bucket["early"] += 1
                if status == "late": bucket["late"] += 1
                if measured:
                    bucket["observed_delays"].append(round(observed, 2))
                    bucket["observed_delays"] = bucket["observed_delays"][-50:]
                bucket["recent"].append({
                    "status": "success" if timing_correct else "loss",
                    "feedback": status,
                    "measured": measured,
                    "actual_delay": round(observed, 2) if measured else None,
                    "offset_seconds": timing_offset_seconds,
                    "confidence": confidence,
                    "time": datetime.datetime.now().isoformat(),
                })
                bucket["recent"] = bucket["recent"][-30:]

            # A +/-1 feedback is a correction hint, not a confirmed success.
            # Keep it separately so similar sequences can weakly bias the next
            # timing choice without falsely counting the adjacent window as won.
            if status in ("early", "late") and selected and timing_offset_seconds in (-60, 60, -60.0, 60.0):
                suggested_delay = selected + int(round(float(timing_offset_seconds) / 60.0))
                if suggested_delay in SUPPORTED_DELAYS:
                    hints = entry.setdefault("timing_suggestions", {})
                    hint = hints.setdefault(str(suggested_delay), {"count": 0, "from_delays": {}, "recent": []})
                    hint["count"] = int(_safe_float(hint.get("count", 0), 0)) + 1
                    from_key = str(selected)
                    hint.setdefault("from_delays", {})[from_key] = int(_safe_float(hint["from_delays"].get(from_key, 0), 0)) + 1
                    hint.setdefault("recent", []).append({"from_delay": selected, "suggested_delay": suggested_delay,
                                                          "offset_seconds": int(timing_offset_seconds),
                                                          "confirmed": False, "time": datetime.datetime.now().isoformat()})
                    hint["recent"] = hint["recent"][-30:]

    # Bound the new derived memory without deleting any legacy memory namespace.
    while len(root) > 10000:
        root.pop(next(iter(root)))


def get_sequence_feature_direction_score(sequence, mode):
    """Retourne une preuve directionnelle prudente; neutre si moins de 3 exemples."""
    root = pattern_stats.get(SEQUENCE_FEATURE_MEMORY_KEY, {}) if isinstance(pattern_stats, dict) else {}
    profile = extract_sequence_features(sequence)
    vals = []
    for label in profile.get("labels", []):
        entry = root.get(f"{mode or 'unknown'}|{label}", {})
        d = entry.get("direction", {}) if isinstance(entry, dict) else {}
        n = int(_safe_float(d.get("total", 0), 0))
        if n < SEQUENCE_FEATURE_MIN_SAMPLES:
            continue
        wins = int(_safe_float(d.get("gain", 0), 0))
        losses = int(_safe_float(d.get("perte", 0), 0))
        if wins + losses:
            rate = (wins + 1.0) / (wins + losses + 2.0)
            vals.append((rate, min(1.0, n / 12.0)))
    if not vals:
        return {"score": 0.5, "confidence": 0.0, "features": profile.get("labels", [])}
    weight = sum(w for _, w in vals)
    score = sum(v * w for v, w in vals) / max(weight, 1e-9)
    return {"score": clamp(score, 0.0, 1.0), "confidence": clamp(weight / len(vals), 0.0, 1.0),
            "features": profile.get("labels", [])}


def get_sequence_feature_timing_score(sequence, mode, delay):
    """Score une fenêtre via résultats mesurés et indices +/-1 non confirmés.

    Les corrections suggérées ont un poids faible et restent distinctes des
    succès mesurés; plusieurs labels issus d'un même feedback ne sont pas
    additionnés comme s'il s'agissait d'observations indépendantes.
    """
    root = pattern_stats.get(SEQUENCE_FEATURE_MEMORY_KEY, {}) if isinstance(pattern_stats, dict) else {}
    profile = extract_sequence_features(sequence)
    try:
        delay = int(delay)
    except Exception:
        return {"score": 0.5, "samples": 0, "suggestion_samples": 0, "effective_samples": 0.0}
    actual_scores = []
    max_actual_support = 0
    max_suggestion_support = 0
    suggestion_scores = []

    for label in profile.get("labels", []):
        entry = root.get(f"{mode or 'unknown'}|{label}", {})
        if not isinstance(entry, dict):
            continue
        bucket = entry.get("timing_by_delay", {}).get(str(delay), {})
        n = int(_safe_float(bucket.get("total", 0), 0))
        if n >= SEQUENCE_FEATURE_MIN_SAMPLES:
            wins = int(_safe_float(bucket.get("success", 0), 0))
            losses = int(_safe_float(bucket.get("loss", 0), 0))
            measured_n = wins + losses
            if measured_n >= SEQUENCE_FEATURE_MIN_SAMPLES:
                actual_scores.append(((wins + 1.0) / (measured_n + 2.0), min(1.0, measured_n / 12.0)))
                max_actual_support = max(max_actual_support, measured_n)

        hints = entry.get("timing_suggestions", {})
        if isinstance(hints, dict):
            counts = {int(k): int(_safe_float(v.get("count", 0), 0)) for k, v in hints.items()
                      if str(k).isdigit() and isinstance(v, dict)}
            total_hints = sum(max(0, v) for v in counts.values())
            if total_hints > 0:
                this_count = max(0, counts.get(delay, 0))
                share = (this_count + 1.0) / (total_hints + float(len(SUPPORTED_DELAYS)))
                hint_score = clamp(0.50 + (share - 1.0 / len(SUPPORTED_DELAYS)) * 0.75, 0.25, 0.75)
                suggestion_scores.append((hint_score, min(3.0, total_hints * 0.5)))
                max_suggestion_support = max(max_suggestion_support, total_hints)

    if actual_scores:
        weight_sum = sum(weight for _, weight in actual_scores)
        actual_score = sum(score * weight for score, weight in actual_scores) / max(weight_sum, 1e-9)
    else:
        actual_score = 0.5

    suggestion_weight = min(3.0, max_suggestion_support * 0.5)
    if suggestion_scores:
        hint_weight_sum = sum(weight for _, weight in suggestion_scores)
        hint_score = sum(score * weight for score, weight in suggestion_scores) / max(hint_weight_sum, 1e-9)
    else:
        hint_score = 0.5

    total_weight = max_actual_support + suggestion_weight
    score = ((actual_score * max_actual_support) + (hint_score * suggestion_weight)) / total_weight if total_weight > 0 else 0.5
    return {"score": clamp(score, 0.0, 1.0), "samples": max_actual_support,
            "suggestion_samples": max_suggestion_support, "effective_samples": total_weight}


def rebuild_sequence_feature_memory():
    """Rebuild uniquement la mémoire dérivée v1 à partir des feedbacks conservés."""
    global pattern_stats
    if not isinstance(pattern_stats, dict): pattern_stats = {}
    rebuilt = {}
    old = pattern_stats.get(SEQUENCE_FEATURE_MEMORY_KEY)
    pattern_stats[SEQUENCE_FEATURE_MEMORY_KEY] = rebuilt
    for record in (feedback_history if isinstance(feedback_history, list) else []):
        if not isinstance(record, dict): continue
        status = str(record.get("learning_status", record.get("status", "")) or "").lower().strip()
        prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
        sequence = record.get("pattern_sequence") or prediction.get("pattern_sequence") or record.get("cotes") or prediction.get("cotes") or []
        if not sequence: continue
        delay = record.get("predicted_delay", record.get("delay_min", prediction.get("delay_min")))
        source = record.get("actual_delay_source")
        actual = record.get("actual_delay_sec")
        if not source and record.get("actual_datetime") and actual is not None:
            source = "observed_datetime"
        update_sequence_feature_memory(sequence, record.get("mode", prediction.get("mode")), status,
                                       predicted_delay=delay, actual_delay=actual,
                                       actual_delay_source=source,
                                       timing_offset_seconds=record.get("timing_offset_seconds"),
                                       confidence=record.get("timing_confidence", prediction.get("timing_confidence", 0.5)))
    logging.info("[SEQUENCE FEATURE MEMORY] %s familles reconstruites", len(rebuilt))
    return rebuilt


def rebuild_pattern_timing_memory():
    """Reconstruit la mémoire temporelle sans utiliser Gain/Perte comme preuve temporelle.

    Gain/Perte ne contribuent que si une heure réelle a été saisie et qu'un délai
    mesuré est disponible. Early/late pénalise seulement le délai sélectionné;
    aucune fenêtre voisine n'est déclarée gagnante sans observation mesurée.
    """
    global pattern_stats
    rebuilt = {}
    records = feedback_history if isinstance(feedback_history, list) else []
    for record in records:
        if not isinstance(record, dict): continue
        status = str(record.get("status", record.get("learning_status", "")) or "").lower().strip()
        if status not in ("gain", "perte", "loss", "early", "late"): continue
        mode = record.get("mode")
        if not mode: continue
        source = record.get("actual_delay_source")
        if not source and record.get("actual_datetime") and record.get("actual_delay_sec") is not None:
            source = "observed_datetime"
        prediction = record.get("prediction", {})
        if not isinstance(prediction, dict): prediction = {}
        pattern = record.get("pattern_sequence") or prediction.get("pattern_sequence") or record.get("cotes") or prediction.get("cotes") or []
        values = list(pattern or [])[-5:]
        if not values: continue
        predicted_delay = _safe_float(record.get("predicted_delay", record.get("delay_min", prediction.get("delay_min"))), 0.0)
        if predicted_delay <= 0: predicted_delay = _safe_float(prediction.get("delay_sec"), 0.0) / 60.0
        predicted_delay = int(round(predicted_delay))
        if predicted_delay not in SUPPORTED_DELAYS: continue
        actual_delay = _safe_float(record.get("actual_delay_sec"), -1.0)
        measured = source == "observed_datetime" and 0 < actual_delay <= 1800
        if status in ("gain", "perte", "loss") and not measured:
            continue
        if status in ("early", "late") and source == "observed_datetime" and not measured:
            # A malformed/invalid observed timestamp cannot be used as evidence.
            continue
        timing_correct = bool(measured and status not in ("early", "late") and _delay_bucket(actual_delay) == predicted_delay and abs(actual_delay - _delay_center(predicted_delay)) <= 30.0)
        keys = []
        exact_key = _pattern_timing_key(values, mode)
        if exact_key: keys.append(exact_key)
        classes = []
        for value in values:
            value = _safe_float(value, 0.0)
            classes.append("L" if value < 1.30 else "M" if value < 2.00 else "H" if value < 5.00 else "X")
        for length in (4, 3):
            if len(classes) >= length:
                sub_key = f"{mode or 'unknown'}|" + "_".join(classes[-length:])
                if sub_key not in keys: keys.append(sub_key)
        confidence = _safe_float(prediction.get("timing_confidence"), _safe_float(record.get("timing_confidence"), 0.50))
        confidence = clamp(confidence, 0.0, 1.0)
        for pattern_key in keys:
            bucket = rebuilt.setdefault(pattern_key, {})
            entry = bucket.setdefault(str(predicted_delay), {"total": 0, "timing_success": 0, "timing_loss": 0,
                "early": 0, "late": 0, "observed_delays": [], "recent_results": [], "confidence_sum": 0.0})
            entry["total"] += 1
            entry["timing_success" if timing_correct else "timing_loss"] += 1
            if status == "early": entry["early"] += 1
            if status == "late": entry["late"] += 1
            if measured:
                entry["observed_delays"].append(actual_delay)
                entry["observed_delays"] = entry["observed_delays"][-50:]
            entry["confidence_sum"] += confidence
            entry["recent_results"].append({"status": "timing_success" if timing_correct else "timing_loss",
                "feedback": status, "directional_status": status if status in ("gain", "perte", "loss") else None,
                "predicted_delay": predicted_delay, "actual_delay": actual_delay if measured else None,
                "actual_delay_source": source, "confidence": confidence, "time": record.get("time", "")})
            entry["recent_results"] = entry["recent_results"][-30:]
    if len(rebuilt) > PATTERN_DELAY_MAX_ENTRIES:
        rebuilt = dict(list(rebuilt.items())[-PATTERN_DELAY_MAX_ENTRIES:])
    if not isinstance(pattern_stats, dict): pattern_stats = {}
    pattern_stats[PATTERN_DELAY_MEMORY_KEY] = rebuilt
    logging.info("[PATTERN TIMING ADAPTIVE] %s familles reconstruites", len(rebuilt))
    return rebuilt


def update_pattern_memory(pattern_seq, result):
    """
    Met à jour la mémoire des patterns après un résultat définitif.

    IMPORTANT :
    - gain/perte modifie la mémoire directionnelle ;
    - early/late ne doit PAS modifier la mémoire gain/perte ;
    - la comparaison est effectuée AVANT l'insertion du nouveau pattern
      afin d'éviter qu'un pattern nouvellement ajouté se compare à lui-même.
    """
    if not pattern_seq:
        return

    result = str(result).lower().strip()

    if result not in ("gain", "perte", "loss"):
        return

    normalized_result = "gain" if result == "gain" else "perte"

    try:
        lr = 0.12

        target_key = "gains" if normalized_result == "gain" else "pertes"
        other_key = "pertes" if target_key == "gains" else "gains"

        # ------------------------------------------------------------------
        # 1. Comparaison AVANT insertion
        # ------------------------------------------------------------------
        similarity_hits = {}

        for L in (5, 10, 15):
            if len(pattern_seq) < L:
                continue

            seq = list(pattern_seq[-L:])

            target_collection = (
                pattern_memory
                .setdefault(target_key, {})
                .setdefault(L, [])
            )

            other_collection = (
                pattern_memory
                .setdefault(other_key, {})
                .setdefault(L, [])
            )

            target_match = False
            target_similarity = 0.0

            other_match = False
            other_similarity = 0.0

            try:
                target_match, target_similarity = is_similar_pattern(
                    seq,
                    target_collection
                )
            except Exception:
                pass

            try:
                other_match, other_similarity = is_similar_pattern(
                    seq,
                    other_collection
                )
            except Exception:
                pass

            similarity_hits[L] = {
                "target_match": bool(target_match),
                "target_similarity": safe_float(target_similarity, 0.0),
                "other_match": bool(other_match),
                "other_similarity": safe_float(other_similarity, 0.0),
            }

        # ------------------------------------------------------------------
        # 2. Enregistrement du nouveau pattern
        # ------------------------------------------------------------------
        _store_pattern_once(pattern_seq, target_key)

        # ------------------------------------------------------------------
        # 3. Adaptation des poids
        # ------------------------------------------------------------------
        for L, info in similarity_hits.items():

            target_match = info["target_match"]
            target_similarity = clamp(
                info["target_similarity"] / 100.0
                if info["target_similarity"] > 1.0
                else info["target_similarity"],
                0.0,
                1.0
            )

            other_match = info["other_match"]
            other_similarity = clamp(
                info["other_similarity"] / 100.0
                if info["other_similarity"] > 1.0
                else info["other_similarity"],
                0.0,
                1.0
            )

            if normalized_result == "gain":

                if target_match:
                    gain_bonus = lr * (
                        0.03 + 0.04 * max(0.0, min(1.0, target_similarity))
                    )
                    SEQ_WEIGHTS[L] = min(
                        0.70,
                        SEQ_WEIGHTS[L] + gain_bonus
                    )

                if other_match and other_similarity > target_similarity:
                    loss_penalty = lr * (
                        0.02 + 0.03 * max(0.0, min(1.0, other_similarity))
                    )
                    SEQ_WEIGHTS[L] = max(
                        0.10,
                        SEQ_WEIGHTS[L] - loss_penalty
                    )

            else:

                if target_match:
                    loss_bonus = lr * (
                        0.02 + 0.03 * max(0.0, min(1.0, target_similarity))
                    )
                    SEQ_WEIGHTS[L] = min(
                        0.70,
                        SEQ_WEIGHTS[L] + loss_bonus
                    )

                if other_match and other_similarity > target_similarity:
                    gain_penalty = lr * (
                        0.02 + 0.03 * max(0.0, min(1.0, other_similarity))
                    )
                    SEQ_WEIGHTS[L] = max(
                        0.10,
                        SEQ_WEIGHTS[L] - gain_penalty
                    )

        _normalize_pattern_weights()

        logging.info(
            "[PATTERN MEMORY] result=%s | "
            "patterns=%s | weights=%s",
            normalized_result,
            {
                L: len(
                    pattern_memory
                    .get(target_key, {})
                    .get(L, [])
                )
                for L in (5, 10, 15)
            },
            {
                L: round(
                    safe_float(SEQ_WEIGHTS.get(L), 0.0),
                    4
                )
                for L in (5, 10, 15)
            }
        )

    except Exception as e:
        logging.warning(f"[PATTERN MEMORY UPDATE] {e}", exc_info=True)

# ══════════════════════════════════════════════════════════════
# RECALCUL DE LA CONFIANCE DE LA PRÉDICTION
# ══════════════════════════════════════════════════════════════

def _v6_percentile(values, quantile):
    values = sorted(float(v) for v in (values or []) if math.isfinite(float(v)))
    if not values:
        return None
    q = clamp(float(quantile), 0.0, 1.0)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _v6_historical_level_ratios(mode, minimum_samples=30):
    """Calibrate multiplier thresholds only when enough same-mode outcomes exist.

    Ratios describe the fraction of the announced target reached by the actual
    maximum cote. They are empirical thresholds, not guaranteed probabilities.
    """
    ratios = []
    for sample in (feedback_samples if isinstance(feedback_samples, list) else []):
        if not isinstance(sample, dict) or str(sample.get("mode", "")) != str(mode):
            continue
        status = str(sample.get("status", sample.get("learning_status", ""))).lower().strip()
        if status not in ("gain", "perte", "loss"):
            continue
        target = _safe_float(sample.get("predicted_target", sample.get("target")), 0.0)
        actual = _safe_float(sample.get("max_cote_reached", sample.get("actual_max_cote", sample.get("max_cote"))), 0.0)
        if target <= 0 or actual <= 0:
            continue
        ratios.append(clamp(actual / target, 0.05, 1.50))
    if len(ratios) < minimum_samples:
        return {"n": len(ratios), "status": "insufficient_data", "assurance_ratio": None, "reliability_ratio": None}
    assurance_ratio = clamp(_v6_percentile(ratios, 0.20), 0.18, 0.82)
    reliability_ratio = clamp(_v6_percentile(ratios, 0.50), 0.30, 0.94)
    if reliability_ratio < assurance_ratio + 0.08:
        reliability_ratio = min(0.94, assurance_ratio + 0.08)
    return {"n": len(ratios), "status": "calibrated", "assurance_ratio": assurance_ratio, "reliability_ratio": reliability_ratio}



def v7_conditional_magnitude_calibration(current_context, mode, minimum_samples=30):
    """Weighted, same-mode magnitude calibration from completed past feedback.

    Uses only feedback samples already resolved before this prediction. Sequence
    similarity and time-of-day proximity weight the observations; small cohorts
    are reported but never allowed to move the target.
    """
    current_context = current_context if isinstance(current_context, dict) else {}
    current_sequence = current_context.get("sequence", []) or []
    now = datetime.datetime.now()
    observations = []
    for sample in (feedback_samples if isinstance(feedback_samples, list) else []):
        if not isinstance(sample, dict) or str(sample.get("mode", "")) != str(mode):
            continue
        status = str(sample.get("status", sample.get("learning_status", ""))).lower().strip()
        if status not in ("gain", "perte", "loss"):
            continue
        predicted = _safe_float(sample.get("predicted_target", sample.get("target")), 0.0)
        actual = _safe_float(sample.get("max_cote_reached", sample.get("actual_max_cote", sample.get("max_cote"))), 0.0)
        if predicted <= 0 or actual <= 0:
            continue
        historical_sequence = sample.get("sequence", []) or []
        sims = []
        for length in (5, 10, 15):
            sim = _historical_sequence_similarity(current_sequence, historical_sequence, length)
            if sim is not None:
                sims.append((length, sim))
        if not sims:
            continue
        weights = {5: 0.25, 10: 0.33, 15: 0.42}
        seq_sim = sum(weights[n] * v for n, v in sims) / sum(weights[n] for n, _ in sims)
        # Pour la proximité horaire, utiliser l'heure de la prédiction passée,
        # pas l'heure à laquelle l'utilisateur a éventuellement saisi le feedback.
        stamp = None
        for stamp_key in ("prediction_datetime", "created_at", "timestamp"):
            stamp_value = sample.get(stamp_key)
            if not stamp_value:
                continue
            try:
                stamp = datetime.datetime.fromisoformat(str(stamp_value).replace("Z", "+00:00"))
                if stamp.tzinfo is not None:
                    stamp = stamp.astimezone().replace(tzinfo=None)
                break
            except Exception:
                stamp = None
        time_weight = 0.65
        if stamp is not None:
            now_min = now.hour * 60 + now.minute
            old_min = stamp.hour * 60 + stamp.minute
            delta = min(abs(now_min-old_min), 1440-abs(now_min-old_min))
            # Smooth weighting, not a hard cutoff; same time of day is most useful.
            time_weight = 0.45 + 0.55 * math.exp(-delta / 180.0)
        recency = v6_recency_weight(sample.get("created_at") or sample.get("prediction_datetime") or "", 60.0)
        weight = clamp(seq_sim * time_weight * recency, 0.0, 1.0)
        if weight < 0.08:
            continue
        observations.append({"ratio": clamp(actual / predicted, 0.05, 1.50), "weight": weight})

    def weighted_quantile(items, quantile):
        ordered = sorted(items, key=lambda x: x["ratio"])
        total = sum(x["weight"] for x in ordered)
        if total <= 0:
            return None
        threshold = total * quantile
        acc = 0.0
        for item in ordered:
            acc += item["weight"]
            if acc >= threshold:
                return item["ratio"]
        return ordered[-1]["ratio"]

    total_weight = sum(x["weight"] for x in observations)
    effective_n = (total_weight ** 2 / sum(x["weight"] ** 2 for x in observations)) if observations else 0.0
    base = {"version": 1, "status": "insufficient_data", "samples": len(observations),
            "effective_samples": round(effective_n, 2), "weight": 0.0,
            "q20_ratio": None, "q50_ratio": None, "q80_ratio": None,
            "target_adjustment": 0.0}
    if effective_n < minimum_samples:
        return base
    q20, q50, q80 = (weighted_quantile(observations, q) for q in (0.20, 0.50, 0.80))
    # A robust median correction, shrunk and tightly bounded to avoid overreacting.
    calibration_weight = min(0.30, 0.10 + (effective_n - minimum_samples) / 200.0)
    adjustment = clamp((q50 - 1.0) * calibration_weight, -0.04, 0.04)
    base.update({"status": "calibrated", "weight": round(calibration_weight, 4),
                 "q20_ratio": round(q20, 4), "q50_ratio": round(q50, 4),
                 "q80_ratio": round(q80, 4), "target_adjustment": round(adjustment, 5)})
    return base


def recompute_target_levels(result, mode, memory_result=None):
    """Calcule assurance/fiabilité à partir des preuves et, si possible, des résultats historiques.

    Les valeurs restent des multiplicateurs-cibles, pas des probabilités. La
    séparation s'élargit quand l'incertitude ou la nouveauté est élevée. Une
    calibration empirique par mode n'est activée qu'à partir de 30 observations.
    """
    result = dict(result or {})
    min_t, max_t = TARGET_RANGES.get(mode, (2.0, 10.0))
    target = clamp(safe_float(result.get("target"), min_t), min_t, max_t)
    confidence = clamp(safe_float(result.get("confidence"), result.get("score", .5)), 0, 1)
    agreement = clamp(safe_float(result.get("model_agreement"), .5), 0, 1)
    evidence = clamp(safe_float(result.get("evidence_confidence"), .5), 0, 1)
    novelty = clamp(safe_float(result.get("novelty_score"), .5), 0, 1)
    magnitude = clamp(safe_float(result.get("magnitude_confidence"), 0), 0, 1)
    memory = clamp(safe_float((memory_result or {}).get("confidence"), 0), 0, 1)

    uncertainty = clamp(.28 * (1-confidence) + .20 * (1-agreement) + .18 * (1-evidence)
                        + .18 * novelty + .10 * (1-magnitude) + .06 * (1-memory), 0, 1)
    # Niveaux réellement différenciés : la fiabilité dépend des preuves et de
    # l'accord des modèles; l'assurance est plus prudente et s'écarte davantage
    # lorsque l'incertitude, la nouveauté ou le manque de preuves augmentent.
    # Ces ratios sont calculés à chaque prédiction, jamais repris comme valeurs fixes.
    spread_ratio = clamp(
        .20 + .28 * uncertainty + .08 * novelty + .05 * (1 - evidence),
        .18, .52
    )
    reliability_ratio = clamp(
        .54 + .24 * confidence + .10 * agreement + .05 * evidence
        - .18 * uncertainty - .10 * novelty + .05 * memory,
        .30, .88
    )
    assurance_ratio = clamp(
        reliability_ratio - spread_ratio,
        .12, max(.12, reliability_ratio - .10)
    )

    historical = _v6_historical_level_ratios(mode)
    historical_weight = 0.0
    if historical.get("status") == "calibrated":
        # Smoothly increase influence as same-mode evidence accumulates; cap at 55%.
        historical_weight = min(.55, .20 + max(0, historical["n"] - 30) / 180.0)
        assurance_ratio = (1-historical_weight)*assurance_ratio + historical_weight*historical["assurance_ratio"]
        reliability_ratio = (1-historical_weight)*reliability_ratio + historical_weight*historical["reliability_ratio"]
        if reliability_ratio < assurance_ratio + .08:
            reliability_ratio = min(.94, assurance_ratio + .08)

    conditional = result.get("conditional_magnitude_calibration") or {}
    if conditional.get("status") == "calibrated":
        conditional_weight = min(0.25, safe_float(conditional.get("weight"), 0.0))
        q20 = safe_float(conditional.get("q20_ratio"), assurance_ratio)
        q50 = safe_float(conditional.get("q50_ratio"), reliability_ratio)
        assurance_ratio = (1.0-conditional_weight)*assurance_ratio + conditional_weight*clamp(q20, 0.18, 0.82)
        reliability_ratio = (1.0-conditional_weight)*reliability_ratio + conditional_weight*clamp(q50, 0.30, 0.94)
        if reliability_ratio < assurance_ratio + 0.08:
            reliability_ratio = min(0.94, assurance_ratio + 0.08)

    assurance = max(1.01, min(target - .25, target * assurance_ratio))
    reliability = max(assurance + .10, min(target - .12, target * reliability_ratio))
    reliability = min(reliability, target - .05)
    assurance = min(assurance, reliability - .10)
    assurance = max(1.01, assurance)

    result["assurance"] = round(clamp(assurance, 1.01, max_t), 2)
    result["fiabilite"] = round(clamp(reliability, result["assurance"] + .10, max_t), 2)
    result["confidence_levels"] = {
        "uncertainty": round(uncertainty, 4), "spread_ratio": round(spread_ratio, 4),
        "assurance_ratio": round(assurance_ratio, 4), "reliability_ratio": round(reliability_ratio, 4),
        "min_gap": round(target - result["fiabilite"], 2), "confidence": round(confidence, 4),
        "agreement": round(agreement, 4), "evidence": round(evidence, 4),
        "memory_confidence": round(memory, 4), "historical_calibration_n": historical.get("n", 0),
        "historical_calibration_status": historical.get("status"), "historical_weight": round(historical_weight, 3),
        "conditional_magnitude_samples": (result.get("conditional_magnitude_calibration") or {}).get("samples", 0),
        "conditional_magnitude_status": (result.get("conditional_magnitude_calibration") or {}).get("status", "unavailable"),
    }
    return result


def recompute_prediction_confidence(
    result,
    cotes,
    mode,
    pattern_result=None,
    memory_result=None,
):
    """
    Recalcule la confiance finale de la prédiction à partir de plusieurs
    sources indépendantes :

    - score IA
    - mémoire historique
    - similarité des patterns
    - accord des modèles
    - preuves temporelles
    - stabilité des cotes

    Cette fonction ne fabrique pas une nouvelle prédiction :
    elle ajuste uniquement les indicateurs de confiance déjà présents.
    """

    try:
        result = dict(result or {})
        cotes = list(cotes or [])

        # ------------------------------------------------------
        # 1. Score IA de base
        # ------------------------------------------------------
        ai_score = safe_float(
            result.get("score"),
            0.50
        )
        ai_score = max(0.0, min(1.0, ai_score))

        # ------------------------------------------------------
        # 2. Mémoire historique
        # ------------------------------------------------------
        memory_score = 0.50
        memory_conf = 0.0

        if isinstance(memory_result, dict):
            memory_score = safe_float(
                memory_result.get("memory_score"),
                0.50
            )
            memory_conf = safe_float(
                memory_result.get("confidence"),
                0.0
            )

        memory_score = max(0.0, min(1.0, memory_score))
        memory_conf = max(0.0, min(1.0, memory_conf))

        # ------------------------------------------------------
        # 3. Similarité pattern
        # ------------------------------------------------------
        pattern_conf = 0.0

        if isinstance(pattern_result, dict):
            possible_values = [
                pattern_result.get("confidence"),
                pattern_result.get("score"),
                pattern_result.get("pattern_score"),
            ]

            vals = []
            for v in possible_values:
                if v is not None:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass

            if vals:
                pattern_conf = sum(vals) / len(vals)

        # Si la valeur est exprimée en %
        if pattern_conf > 1.0:
            pattern_conf /= 100.0

        pattern_conf = max(0.0, min(1.0, pattern_conf))

        # ------------------------------------------------------
        # 4. Stabilité des données
        # ------------------------------------------------------
        stability = safe_float(
            result.get("stability"),
            0.50
        )

        if stability > 1.0:
            stability /= 100.0

        stability = max(0.0, min(1.0, stability))

        # ------------------------------------------------------
        # 5. Entropie
        # ------------------------------------------------------
        entropy = safe_float(
            result.get("entropy"),
            0.50
        )

        # Une entropie élevée signifie généralement plus
        # d'incertitude.
        if entropy > 1.0:
            entropy /= 100.0

        entropy = max(0.0, min(1.0, entropy))

        entropy_conf = 1.0 - entropy

        # ------------------------------------------------------
        # 6. Evidence timing
        # ------------------------------------------------------
        evidence_conf = safe_float(
            result.get("evidence_confidence"),
            0.50
        )

        if evidence_conf > 1.0:
            evidence_conf /= 100.0

        evidence_conf = max(0.0, min(1.0, evidence_conf))

        # ------------------------------------------------------
        # 7. Accord des modèles
        # ------------------------------------------------------
        agreement = safe_float(
            result.get("model_agreement"),
            0.50
        )

        if agreement > 1.0:
            agreement /= 100.0

        agreement = max(0.0, min(1.0, agreement))

        # ------------------------------------------------------
        # 8. Calcul pondéré
        # ------------------------------------------------------
        #
        # La mémoire n'est importante que lorsqu'elle possède
        # réellement suffisamment de confiance.
        #
        effective_memory = (
            memory_score * memory_conf
            + 0.50 * (1.0 - memory_conf)
        )

        ensemble_score = _v6_norm_signal(result.get("ensemble_score"), ai_score)
        ensemble_confidence = 1.0 - _v6_norm_signal(result.get("ensemble_uncertainty"), 0.50)
        confidence = (
            ai_score * 0.22
            + ensemble_score * 0.18
            + effective_memory * 0.14
            + pattern_conf * 0.12
            + stability * 0.11
            + entropy_conf * 0.08
            + evidence_conf * 0.09
            + agreement * 0.06
        )
        confidence *= (0.82 + 0.18 * ensemble_confidence)

        confidence = max(0.0, min(1.0, confidence))

        # ------------------------------------------------------
        # 9. Pénalité si trop peu de données
        # ------------------------------------------------------
        sample_factor = min(1.0, len(cotes) / 15.0)

        # On évite qu'une analyse avec seulement 5 cotes
        # affiche artificiellement une confiance énorme.
        confidence = (
            confidence * sample_factor
            + 0.50 * (1.0 - sample_factor)
        )

        confidence = max(0.0, min(1.0, confidence))

        # ------------------------------------------------------
        # 10. Composants détaillés
        # ------------------------------------------------------
        result["confidence_components"] = {
            "ai_score": round(ai_score, 4),
            "memory_score": round(memory_score, 4),
            "memory_confidence": round(memory_conf, 4),
            "pattern_confidence": round(pattern_conf, 4),
            "stability": round(stability, 4),
            "entropy_confidence": round(entropy_conf, 4),
            "evidence_confidence": round(evidence_conf, 4),
            "model_agreement": round(agreement, 4),
            "sample_factor": round(sample_factor, 4),
        }

        # ------------------------------------------------------
        # 11. Indicateurs affichés
        # ------------------------------------------------------
        result["evidence_confidence"] = round(
            (
                evidence_conf * 0.50
                + pattern_conf * 0.25
                + memory_conf * 0.25
            ),
            4
        )

        result["model_agreement"] = round(
            (
                agreement * 0.50
                + ai_score * 0.25
                + stability * 0.25
            ),
            4
        )

        result["confidence"] = round(confidence, 4)

        # Pertinence reste exprimée en %
        result["pertinence"] = round(confidence * 100.0, 2)

        return result

    except Exception as e:
        logging.warning(
            f"[CONFIDENCE] recalcul impossible : {e}"
        )

        # Fallback sécurisé : on conserve le résultat existant
        result = dict(result or {})

        existing_score = safe_float(
            result.get("score"),
            0.50
        )

        existing_score = max(
            0.0,
            min(1.0, existing_score)
        )

        result["confidence"] = existing_score
        result["pertinence"] = round(
            existing_score * 100.0,
            2
        )

        result["evidence_confidence"] = existing_score
        result["model_agreement"] = existing_score

        return result



# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_image(im: Image.Image, target_width=1400) -> Image.Image:
    """
    Prétraitement OCR plus robuste.

    On conserve une version noir/blanc nette tout en agrandissant les petites
    cotes afin d'améliorer la reconnaissance Tesseract.
    """
    try:
        if im is None:
            return im

        im = im.convert("L")

        # Agrandissement AVANT certains traitements pour les petits chiffres.
        w, h = im.size

        if w > 0 and w < target_width:
            ratio = target_width / float(w)
            new_h = max(1, int(h * ratio))
            im = im.resize(
                (target_width, new_h),
                Image.Resampling.LANCZOS
            )

        im = ImageOps.autocontrast(im)

        # Réduction du bruit
        im = im.filter(ImageFilter.MedianFilter(size=3))

        # Accentuation
        im = ImageEnhance.Sharpness(im).enhance(1.8)

        # Contraste léger
        im = ImageEnhance.Contrast(im).enhance(1.25)

        # Binarisation : le seuil de 190 conserve les caractères blancs sur
        # les tuiles orange plus lumineuses après le renforcement du contraste.
        im = im.point(
            lambda x: 0 if x < 190 else 255
        )

        return im

    except Exception as e:
        logging.warning(f"[OCR PREPROCESS] {e}")
        return im


def _clean_ocr_number(raw_value):
    """
    Transforme différentes sorties OCR en float exploitable.
    """
    try:
        if raw_value is None:
            return None

        original = str(raw_value).strip()

        if not original:
            return None

        # Corrections OCR fréquentes
        cleaned = original

        replacements = {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        cleaned = cleaned.replace(",", ".")

        # Ne conserver que chiffres et point
        cleaned = re.sub(r"[^\d.]", "", cleaned)

        if not cleaned:
            return None

        # Plusieurs points => essayer de réparer
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])

        value = float(cleaned)

        # Correction de certains résultats OCR sans décimale.
        #
        # Exemple :
        # "123" peut correspondre à 1.23x
        # "734" peut correspondre à 7.34x
        #
        # On n'applique cette correction que pour des valeurs plausibles.
        if (
            100 <= value < 1000
            and "." not in original
        ):
            value /= 100.0

        if not math.isfinite(value):
            return None

        if not (1.0 <= value <= 1000.0):
            return None

        return round(value, 3)

    except Exception:
        return None


def _parse_ocr_pass(raw_text):
    """
    Extrait les valeurs dans l'ordre d'un seul passage OCR.

    Les valeurs explicites avec x/× sont prioritaires. Le repli numérique
    n'est utilisé que si ce passage ne contient aucune cote marquée.
    """
    explicit_tokens = [
        match.group(1)
        for match in re.finditer(
            r"(?<![\d.])(\d{1,4}(?:[.,]\d+)?)\s*[xX×]",
            raw_text or ""
        )
    ]

    tokens = explicit_tokens
    explicit = bool(explicit_tokens)

    if not tokens:
        tokens = re.findall(
            r"(?<![\d.])(\d{1,4}(?:[.,]\d+)?)(?!\d)",
            raw_text or ""
        )

    values = []
    for token in tokens:
        value = _clean_ocr_number(token)
        if value is not None:
            values.append(value)

    return values, explicit


def extract_cotes(file_path, debug=False):
    """
    Exécute plusieurs configurations OCR, puis retient un seul passage.

    Ne jamais concaténer les passages : ils lisent souvent les mêmes cotes,
    ce qui doublait le total. Les valeurs répétées dans une même capture sont
    conservées et l'ordre du passage sélectionné est préservé.
    """
    try:
        if not file_path or not os.path.exists(file_path):
            logging.warning("[OCR] fichier inexistant")
            return []

        with Image.open(file_path) as original_im:
            im = _preprocess_image(original_im)

        configs = [
            r"--psm 6 -c tessedit_char_whitelist=0123456789.,xX×",
            r"--psm 11 -c tessedit_char_whitelist=0123456789.,xX×",
            r"--psm 7 -c tessedit_char_whitelist=0123456789.,xX×",
        ]

        candidates = []

        for pass_index, cfg in enumerate(configs):
            try:
                raw = pytesseract.image_to_string(
                    im,
                    config=cfg,
                    lang="eng"
                )
                values, explicit = _parse_ocr_pass(raw)
                if values:
                    candidates.append({
                        "index": pass_index,
                        "config": cfg,
                        "values": values,
                        "explicit": explicit,
                    })
            except Exception as e:
                logging.warning("[OCR PASS] %s -> %s", cfg, e)

        if not candidates:
            return []

        # Ne pas mélanger un passage explicite avec un passage de repli.
        explicit_candidates = [
            candidate for candidate in candidates
            if candidate["explicit"]
        ]
        pool = explicit_candidates or candidates

        counts = [len(candidate["values"]) for candidate in pool]
        sorted_counts = sorted(counts)
        middle = len(sorted_counts) // 2
        median_count = (
            sorted_counts[middle]
            if len(sorted_counts) % 2
            else (sorted_counts[middle - 1] + sorted_counts[middle]) / 2
        )

        # En cas de désaccord, privilégier le nombre de cotes le plus souvent
        # reconnu, puis le passage le plus proche de la médiane. Les ex æquo
        # restent déterministes et favorisent le passage le plus court.
        selected = min(
            pool,
            key=lambda candidate: (
                -sum(
                    count == len(candidate["values"])
                    for count in counts
                ),
                abs(len(candidate["values"]) - median_count),
                len(candidate["values"]),
                candidate["index"],
            )
        )
        cotes = selected["values"]

        if debug:
            logging.info(
                "[OCR] passage=%s explicite=%s cotes=%d : %s",
                selected["config"],
                selected["explicit"],
                len(cotes),
                cotes[:30]
            )

        return cotes

    except Exception as e:
        logging.warning(
            f"[OCR ERROR] {e}",
            exc_info=True
        )
        return []


# ══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def is_duplicate_message(chat_id, text) -> bool:
    """
    Empêche Telegram de traiter deux fois le même message dans une courte
    fenêtre temporelle.
    """
    try:
        if text is None:
            return False

        text = str(text).strip()

        if not text:
            return False

        h = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        now = time.time()

        message_hashes.setdefault(
            chat_id,
            {}
        )

        message_hashes[chat_id] = {
            key: timestamp
            for key, timestamp
            in message_hashes[chat_id].items()
            if now - timestamp < MESSAGE_DEDUP_WINDOW
        }

        if h in message_hashes[chat_id]:
            return True

        message_hashes[chat_id][h] = now

        return False

    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# UI KEYBOARDS
# ══════════════════════════════════════════════════════════════════════════════

def main_keyboard():
    m = ReplyKeyboardMarkup(
        resize_keyboard=True
    )

    m.row(
        KeyboardButton("🚀 Grosse Cote"),
        KeyboardButton("🔮 Ultra Grosse Cote")
    )

    m.row(
        KeyboardButton("🎯 2X-10X"),
        KeyboardButton("📜 Historique")
    )

    m.row(
        KeyboardButton("📊 Stats IA"),
        KeyboardButton("🔄 Réentraînement IA")
    )

    m.row(
        KeyboardButton("📝 Observer >10x"),
        KeyboardButton("📋 Heures >10x")
    )

    return m


def feedback_keyboard(prediction_id=None):
    m = InlineKeyboardMarkup()
    suffix = (
        f":{prediction_id}"
        if prediction_id
        else ""
    )

    m.row(
        InlineKeyboardButton(
            "✅ Gain",
            callback_data=f"fb_gain{suffix}"
        ),
        InlineKeyboardButton(
            "❌ Perte",
            callback_data=f"fb_perte{suffix}"
        ),
    )

    m.row(
        InlineKeyboardButton(
            "⏪ -1 min",
            callback_data=f"fb_early{suffix}"
        ),
        InlineKeyboardButton(
            "⏩ +1 min",
            callback_data=f"fb_late{suffix}"
        ),
    )

    return m


# ══════════════════════════════════════════════════════════════════════════════
# SAFE TELEGRAM SEND
# ══════════════════════════════════════════════════════════════════════════════

def _safe_send_message(chat_id, text, **kwargs):
    """
    Wrapper Telegram afin qu'une erreur d'envoi ne fasse pas tomber
    toute la logique de prédiction.
    """
    try:
        return bot.send_message(
            chat_id,
            text,
            **kwargs
        )
    except Exception as e:
        logging.error(
            f"[TELEGRAM SEND] {e}",
            exc_info=True
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def _apply_similar_memory_timing(result: dict, memory_result: dict) -> dict:
    """Blend observed delays from similar, verified experiences into the live timing.

    This is deliberately bounded and activates only after at least three measured
    elapsed-time observations. Qualitative +/-1 minute feedback is not treated as
    an exact timestamp. The original timing model remains the dominant source.
    """
    result = dict(result or {})
    memory_result = memory_result or {}
    try:
        timing_samples = max(0, int(_safe_float(memory_result.get("timing_samples"), 0)))
        if timing_samples < 3:
            result["similar_timing_samples"] = timing_samples
            result["similar_timing_blend"] = 0.0
            return result

        raw_base = result.get("delay_probabilities") or {}
        base = {}
        for delay in SUPPORTED_DELAYS:
            value = raw_base.get(str(delay), raw_base.get(delay))
            if value is None:
                value = (result.get("delay_scores") or {}).get(delay, {}).get("final_score", 0.0)
            base[delay] = max(0.0, _safe_float(value, 0.0))
        base_total = sum(base.values())
        if base_total <= 0:
            base = {delay: 1.0 / len(SUPPORTED_DELAYS) for delay in SUPPORTED_DELAYS}
        else:
            base = {delay: value / base_total for delay, value in base.items()}

        raw_hist = memory_result.get("delay_distribution") or {}
        hist = {delay: max(0.0, _safe_float(raw_hist.get(delay, raw_hist.get(str(delay), 0.0)), 0.0)) for delay in SUPPORTED_DELAYS}
        hist_total = sum(hist.values())
        if hist_total <= 0:
            result["similar_timing_samples"] = timing_samples
            result["similar_timing_blend"] = 0.0
            return result
        hist = {delay: value / hist_total for delay, value in hist.items()}

        timing_confidence = clamp(_safe_float(memory_result.get("timing_confidence"), 0.0), 0.0, 1.0)
        # A small, gradual influence: capped at 25%, even with a large history.
        blend = min(0.25, 0.05 + 0.02 * max(0, timing_samples - 3)) * timing_confidence
        if blend <= 0:
            result["similar_timing_samples"] = timing_samples
            result["similar_timing_blend"] = 0.0
            return result

        combined = {
            delay: (1.0 - blend) * base[delay] + blend * hist[delay]
            for delay in SUPPORTED_DELAYS
        }
        norm = sum(combined.values()) or 1.0
        combined = {delay: value / norm for delay, value in combined.items()}
        # Final memory-adjusted choice is deterministic: the historical blend
        # is applied once, then the strongest combined probability wins.
        # The initial timing model may still use its existing uncertainty-aware
        # exploration when there is not enough verified similar-memory evidence.
        selected = int(max(
            SUPPORTED_DELAYS,
            key=lambda delay: (combined[delay], base[delay], -delay),
        ))

        now = datetime.datetime.now()
        delay_seconds = int(_delay_center(selected))
        window_start = now + datetime.timedelta(minutes=selected)
        window_end = window_start + datetime.timedelta(seconds=60)
        optimal = now + datetime.timedelta(seconds=delay_seconds)

        result["delay_min"] = selected
        result["delay_sec"] = delay_seconds
        result["optimal_time"] = optimal.strftime("%H:%M:%S")
        result["fenetre"] = (window_start.strftime("%H:%M:%S"), window_end.strftime("%H:%M:%S"))
        result["delay_probabilities"] = {str(delay): round(combined[delay], 4) for delay in SUPPORTED_DELAYS}
        result["similar_timing_samples"] = timing_samples
        result["similar_timing_blend"] = round(blend, 4)
        result["similar_timing_distribution"] = {str(delay): round(hist[delay], 4) for delay in SUPPORTED_DELAYS}
        logging.info("[SIMILAR MEMORY TIMING] samples=%s confidence=%.3f blend=%.3f base=%s memory=%s combined=%s selected=%s", timing_samples, timing_confidence, blend, {d: round(base[d], 4) for d in SUPPORTED_DELAYS}, {d: round(hist[d], 4) for d in SUPPORTED_DELAYS}, {d: round(combined[d], 4) for d in SUPPORTED_DELAYS}, selected)
        return result
    except Exception as exc:
        logging.warning("[SIMILAR MEMORY TIMING] recalibration skipped: %s", exc, exc_info=True)
        return result


def record_final_timing_decision(result: dict) -> None:
    """Update the timing audit with the final delay after memory blending."""
    try:
        if not isinstance(result, dict):
            return
        audit = AI_STATE.setdefault("timing_selection_audit", {})
        recent = audit.setdefault("recent", [])
        if not recent:
            return
        probabilities = result.get("delay_probabilities") or {}
        final_delay = int(_safe_float(result.get("delay_min"), 0))
        if final_delay not in SUPPORTED_DELAYS:
            return
        # Update the most recent live decision; this runs immediately after
        # the memory adjustment in the same prediction pipeline.
        item = recent[-1]
        item["delay"] = final_delay
        item["final_probabilities"] = {
            str(d): round(_safe_float(probabilities.get(str(d), probabilities.get(d, 0.0))), 4)
            for d in SUPPORTED_DELAYS
        }
        item["similar_timing_samples"] = int(_safe_float(result.get("similar_timing_samples"), 0))
        item["similar_timing_blend"] = round(_safe_float(result.get("similar_timing_blend"), 0.0), 4)
        item["finalized"] = True
        audit["recent"] = recent[-120:]
    except Exception as exc:
        logging.warning("[TIMING FINAL AUDIT] skipped: %s", exc)


def _handle_mode(msg, mode: str):

    chat_id = msg.chat.id

    try:

        raw_cotes = last_cotes.get(chat_id)

        if not raw_cotes:
            _safe_send_message(
                chat_id,
                "❌ Envoyez d'abord des cotes ou une capture."
            )
            return

        normal_cotes, isolated_high = split_analysis_cotes(
            raw_cotes
        )

        # Les cotes >10x restent visibles dans l'extraction et l'historique,
        # mais sont exclues du calcul pour tous les modes, y compris ULTRA.
        cotes = list(normal_cotes)

        if not cotes:
            _safe_send_message(
                chat_id,
                "❌ Aucune cote exploitable."
            )
            return

        if mode == "ultra":

            ok, msg_f = filtre_ultra(cotes)

            if not ok:
                _safe_send_message(
                    chat_id,
                    f"🔮 <b>Mode ULTRA</b>\n"
                    f"{html_mod.escape(str(msg_f))}\n\n"
                    f"Essayez <b>Grosse Cote</b>.",
                    parse_mode="HTML"
                )
                return

        # ══════════════════════════════════════════════════════════════════
        # 1. PATTERN + SCORE IA
        # ══════════════════════════════════════════════════════════════════

        analysis_slice = cotes[-ANALYSE_LAST:]

        pat = detect_weighted_patterns(
            analysis_slice,
            pattern_memory,
            SEQ_WEIGHTS
        )

        ai_score = compute_ai_score(
            cotes=cotes,
            pattern_data=pattern_memory,
            seq_weights=SEQ_WEIGHTS,
            mode=mode,
        )

        # ══════════════════════════════════════════════════════════════════
        # 2. PREDICTION INITIALE
        # ══════════════════════════════════════════════════════════════════

        result = PRIMARY_ENGINE["run_prediction"](
            cotes,
            mode,
            ai_score,
            pat
        )

        if not result:
            _safe_send_message(
                chat_id,
                "❌ Analyse impossible."
            )
            return

        # ══════════════════════════════════════════════════════════════════
        # 3. MÉMOIRE D'EXPÉRIENCE
        # ══════════════════════════════════════════════════════════════════

        temp_exp = PRIMARY_ENGINE["build_current_experience"](
            cotes,
            mode,
            ai_score
        )

        memory_result = (
            PRIMARY_ENGINE["compute_memory_learning"](temp_exp)
            if temp_exp
            else {}
        )

        # Comparaison historique indépendante : diagnostic uniquement à cette phase.
        # Les cohorts temporelles et les séquences 5/10/15 ne modifient pas encore
        # les scores, la cible ou le délai annoncés.
        try:
            historical_comparison = historical_comparison_engine(temp_exp)
        except Exception as history_exc:
            logging.warning("[HISTORICAL COMPARISON] %s", history_exc)
            historical_comparison = {"version": 1, "status": "error"}
        result["historical_comparison"] = historical_comparison

        # Les temps réellement mesurés dans des expériences similaires peuvent
        # ajuster légèrement le choix 1/2/3 min, avant la construction du message.
        # Le moteur temporel principal reste majoritaire et les échantillons rares
        # ne changent pas le délai annoncé.
        result = _apply_similar_memory_timing(result, memory_result)
        record_final_timing_decision(result)

        # ══════════════════════════════════════════════════════════════════
        # 4. AJUSTEMENT DU SCORE PAR MÉMOIRE
        # ══════════════════════════════════════════════════════════════════

        final_score = apply_memory_learning(
            ai_score,
            memory_result
        )

        # Les parcours historiques de pertes <1.99x associés aux patterns
        # similaires deviennent un signal de prudence. L'influence est graduelle
        # et limitée ; une poignée de retours ne peut pas écraser le moteur.
        low_loss_risk = memory_result.get("low_loss_risk")
        memory_samples = max(0, int(safe_float(memory_result.get("samples"), 0)))
        if low_loss_risk is not None and memory_samples > 0:
            risk_strength = clamp(memory_samples / 8.0, 0.0, 1.0)
            loss_penalty = min(0.08, 0.08 * clamp(safe_float(low_loss_risk), 0.0, 1.0) * risk_strength)
            final_score = clamp(final_score - loss_penalty, 0.0, 1.0)
            result["similar_low_loss_risk"] = round(safe_float(low_loss_risk), 4)
            result["similar_low_loss_samples"] = memory_samples
            result["similar_low_loss_penalty"] = round(loss_penalty, 4)
        else:
            result["similar_low_loss_risk"] = None
            result["similar_low_loss_samples"] = memory_samples
            result["similar_low_loss_penalty"] = 0.0

        final_score = max(
            0.0,
            min(1.0, safe_float(final_score, ai_score))
        )

        result["score"] = round(
            final_score,
            4
        )

        result["pertinence"] = round(final_score * 100, 2)

        ensemble = v6_ensemble_decision(cotes, mode, final_score, pat, memory_result)
        ensemble_score = safe_float(ensemble.get("score"), final_score)
        blend = 0.38 if ensemble.get("feedback_samples", 0) >= V6_ENSEMBLE_MIN_SAMPLES else 0.25
        final_score = clamp((1.0-blend)*final_score + blend*ensemble_score, 0.0, 1.0)
        result["score"] = round(final_score,4)
        result["pertinence"] = round(final_score*100.0,2)
        result["model_agreement"] = ensemble.get("agreement",.5)
        result["evidence_confidence"] = ensemble.get("evidence",.5)
        result["novelty_score"] = ensemble.get("novelty",.5)
        result["ensemble_uncertainty"] = ensemble.get("uncertainty",.5)
        result["ensemble_score"] = ensemble_score
        result["ensemble_weights"] = ensemble.get("weights",{})
        result["ensemble_signals"] = ensemble.get("signals",{})
        result["ensemble_feedback_samples"] = ensemble.get("feedback_samples",0)

        # Ajustement de magnitude très modéré : le consensus peut déplacer
        # la cible, mais jamais sortir de la plage du mode ni la multiplier
        # brutalement.
        target_before_ensemble = safe_float(result.get("target"), TARGET_RANGES.get(mode,(2.0,10.0))[0])
        target_min, target_max = TARGET_RANGES.get(mode,(2.0,10.0))
        target_shift = clamp((ensemble_score - 0.50) * 0.10, -0.05, 0.05)
        result["target"] = round(clamp(target_before_ensemble * (1.0 + target_shift), target_min, target_max), 2)
        result["ensemble_target_shift"] = round(target_shift, 5)

        # Calibration empirique douce : si des patterns proches ont souvent
        # atteint moins/plus que leur cible, on en tient compte ici. Le poids
        # dépend du nombre et de la confiance des expériences similaires.
        historical_target_error = memory_result.get("avg_target_error")
        if historical_target_error is not None and memory_result.get("available"):
            hist_conf = clamp(safe_float(memory_result.get("confidence"), 0.0), 0.0, 1.0)
            hist_samples = max(0, int(safe_float(memory_result.get("samples"), 0)))
            sample_strength = clamp(hist_samples / 10.0, 0.0, 1.0)
            correction = clamp(
                safe_float(historical_target_error) * 0.20 * hist_conf * sample_strength,
                -0.08 * result["target"],
                0.05 * result["target"],
            )
            result["target"] = round(clamp(result["target"] + correction, target_min, target_max), 2)
            result["similar_target_error"] = round(safe_float(historical_target_error), 3)
            result["similar_target_correction"] = round(correction, 3)
        else:
            result["similar_target_error"] = None
            result["similar_target_correction"] = 0.0

        # Calibration conditionnelle de magnitude : elle n'agit que si un
        # historique de même mode et de séquences proches est suffisamment fourni.
        try:
            conditional_magnitude = v7_conditional_magnitude_calibration(
                temp_exp, mode, minimum_samples=30
            )
            result["conditional_magnitude_calibration"] = conditional_magnitude
            if conditional_magnitude.get("status") == "calibrated":
                target_before_calibration = safe_float(result.get("target"), target_min)
                target_adjustment = clamp(
                    target_before_calibration * safe_float(conditional_magnitude.get("target_adjustment"), 0.0),
                    -0.04 * target_before_calibration,
                    0.04 * target_before_calibration,
                )
                result["target"] = round(clamp(target_before_calibration + target_adjustment, target_min, target_max), 2)
                result["conditional_target_adjustment"] = round(target_adjustment, 3)
            else:
                result["conditional_target_adjustment"] = 0.0
        except Exception as calibration_exc:
            logging.warning("[V7 MAGNITUDE CALIBRATION] %s", calibration_exc)
            result["conditional_magnitude_calibration"] = {"version": 1, "status": "error"}
            result["conditional_target_adjustment"] = 0.0

        result = recompute_prediction_confidence(
            result=result,
            cotes=cotes,
            mode=mode,
            pattern_result=pat,
            memory_result=memory_result,
        )

        result = recompute_target_levels(
            result=result,
            mode=mode,
            memory_result=memory_result,
        )

        # Final independent consistency gate: bounded score correction and
        # conservative confidence penalty when model signals disagree.
        result = v7_independent_final_decision(result, memory_result)
        result = recompute_target_levels(
            result=result,
            mode=mode,
            memory_result=memory_result,
        )

        confidence_value = safe_float(
            result.get("confidence", result.get("score", 0.5)), 0.5
        )
        # No abstention: confidence is informational only.
        result["decision_status"] = "PREDICT"
        result["abstention_reason"] = None

        # ══════════════════════════════════════════════════════════════════
        # 5. CONSTRUCTION DE L'EXPÉRIENCE
        # ══════════════════════════════════════════════════════════════════

        pred_time = datetime.datetime.now()

        try:

            last_slice_for_nn = (
                cotes[-ANALYSE_LAST:]
                if cotes
                else []
            )

            nn_features = build_prediction_features(
                consecutive_low_value=float(
                    result.get(
                        "serie_creuse",
                        0
                    )
                ),

                volatility_value=(
                    volatility(last_slice_for_nn)
                    if last_slice_for_nn
                    else 0.0
                ),

                std_value=float(
                    result.get(
                        "ecart",
                        0
                    )
                ),

                mean_value=float(
                    result.get(
                        "moyenne",
                        0
                    )
                ),

                predicted_score=0.0,
                predicted_target=0.0,

                predicted_hour=pred_time.hour,

                predicted_minute=pred_time.minute,

                peak_interval=(
                    compute_peak_interval(
                        last_slice_for_nn
                    )
                    or 0.0
                ),

                sequence_length=len(
                    last_slice_for_nn
                ),

                pattern_gain=0.0,

                pattern_loss=0.0,
            )

            nn_val = predict_with_model(
                nn_features
            )

        except Exception as e:

            logging.warning(
                f"[NN FEATURE] {e}"
            )

            nn_val = 0.5

        # ══════════════════════════════════════════════════════════════════
        # 6. STOCKAGE EXPÉRIENCE
        # ══════════════════════════════════════════════════════════════════

        experience = create_experience_memory(
            cotes=cotes,
            mode=mode,
            ai_score=final_score,
            prediction_result=result,
            markov_confidence=0.0,
            neural_score_val=nn_val,
            neural_conf_val=neural_confidence(),
        )

        if experience:
            store_experience(
                experience
            )

        # ══════════════════════════════════════════════════════════════════
        # 7. CONTEXTE EXPÉRIENCE
        # ══════════════════════════════════════════════════════════════════

        try:

            exp_ctx = get_experience_context(
                cotes,
                mode,
                final_score
            )

            agg = exp_ctx.get(
                "aggregate",
                {}
            )

        except Exception as e:

            logging.warning(
                f"[EXP_CTX] {e}"
            )

            agg = {}

        # ══════════════════════════════════════════════════════════════════
        # 8. MÉMOIRE HISTORIQUE DE LA SITUATION
        # ══════════════════════════════════════════════════════════════════

        try:

            mem_ctx = memory_based_prediction(
                cotes,
                mode,
                final_score
            )

        except Exception as e:

            logging.warning(
                f"[MEM_CTX] {e}"
            )

            mem_ctx = {
                "probability": 0.5,
                "confidence": 0.0,
                "weight": 0.0,
                "samples": 0,
                "active": False,
            }

        last_slice = cotes[-ANALYSE_LAST:]

        # ══════════════════════════════════════════════════════════════════
        # 9. STOCKAGE PENDING PREDICTION
        # ══════════════════════════════════════════════════════════════════

        prediction_id = uuid.uuid4().hex

        pending_prediction = {
            "prediction_id": prediction_id,

            "result": result,

            "cotes": list(cotes),

            "mode": mode,

            "ai_score": final_score,

            "pred_time": pred_time.isoformat(),

            "experience_id": (
                experience.get("experience_id")
                if experience
                else None
            ),

            "model_outputs": (
                _last_prediction_meta.get(
                    "model_outputs",
                    {}
                )
            ),

            "meta_context": (
                _last_prediction_meta.get(
                    "context",
                    "CALM"
                )
            ),

            "prediction_context": {
                "memory": mem_ctx,

                "mean": mean_std(
                    last_slice
                )[0],

                "std": mean_std(
                    last_slice
                )[1],

                "low_series": consecutive_low(
                    last_slice
                ),

                "markov": weighted_markov_next(
                    last_slice
                ),
            },

            # Informations supplémentaires utiles au learning
            "created_at": time.time(),

            "isolated_high_count": len(
                isolated_high
            ),
        }

        pending_predictions.setdefault(
            chat_id,
            {}
        )[prediction_id] = pending_prediction

        # ══════════════════════════════════════════════════════════════════
        # 10. CONSTRUCTION AFFICHAGE
        # ══════════════════════════════════════════════════════════════════

        header = "━━━━━━━━━━━━━━━━━━"

        # ------------------------------------------------------------------
        # Patterns
        # ------------------------------------------------------------------

        pat_txt = ""

        try:

            details_map = result.get(
                "pattern",
                {}
            ).get(
                "details",
                {}
            )

            for L in sorted(
                SEQ_WEIGHTS.keys()
            ):

                d = details_map.get(
                    L,
                    {}
                )

                if (
                    d.get("available")
                    and d.get("gain_match")
                ):

                    pat_txt += (
                        f"• Seq{L}: GAIN "
                        f"{d.get('gain_sim', 0)}%\n"
                    )

            if pat_txt:

                pat_txt = (
                    "🔎 <b>Motifs détectés :</b>\n"
                    + pat_txt
                    + "\n"
                )

        except Exception:
            pat_txt = ""

        # ------------------------------------------------------------------
        # Mémoire
        # ------------------------------------------------------------------

        memory_info = ""

        if memory_result.get(
            "available"
        ):

            memory_info = (
                f"\n🧠 Expériences similaires : "
                f"{memory_result.get('samples', 0)}"
                f"\n📚 Score historique : "
                f"{safe_float(memory_result.get('memory_score'), 0.5) * 100:.1f}%"
                f"\n🔍 Confiance mémoire : "
                f"{safe_float(memory_result.get('confidence'), 0.0) * 100:.1f}%"
            )

        elif agg.get(
            "count",
            0
        ) > 0:

            memory_info = (
                f"\n🧠 Expériences similaires : "
                f"{agg.get('count', 0)}"
                f"\n📚 Taux gain historique : "
                f"{safe_float(agg.get('weighted_gain_rate'), 0.0) * 100:.1f}%"
            )

        # ------------------------------------------------------------------
        # Mémoire situationnelle
        # ------------------------------------------------------------------

        mem_prob_pct = (
            safe_float(
                mem_ctx.get(
                    "probability",
                    0.5
                ),
                0.5
            ) * 100
        )

        mem_samp = int(
            safe_float(
                mem_ctx.get(
                    "samples",
                    0
                ),
                0
            )
        )

        mem_conf_pct = (
            safe_float(
                mem_ctx.get(
                    "confidence",
                    0.0
                ),
                0.0
            ) * 100
        )

        mem_line = ""

        if mem_samp > 0:

            mem_line = (
                f"\n\n🧠 <b>Mémoire historique :</b> "
                f"{mem_prob_pct:.1f}%"
                f"\n📚 Situations similaires : "
                f"{mem_samp}"
                f"\n🔍 Confiance mémoire : "
                f"{mem_conf_pct:.1f}%"
            )

        # ------------------------------------------------------------------
        # Path risk
        # ------------------------------------------------------------------

        path_risk_pct = (
            safe_float(
                result.get(
                    "path_risk",
                    0.0
                ),
                0.0
            ) * 100
        )

        avg_path_losses = (
            result.get(
                "confidence_components",
                {}
            ).get(
                "avg_low_cote_losses"
            )
        )

        path_line = ""

        if avg_path_losses is not None:

            path_line = (
                f"\n🧭 Pertes ≤ 1.99x avant issue similaire : "
                f"{safe_float(avg_path_losses, 0.0):.1f}"
                f"\n⚠️ Risque de parcours : "
                f"{path_risk_pct:.1f}%"
            )

        # ------------------------------------------------------------------
        # Timing
        # ------------------------------------------------------------------

        timing_scores = result.get(
            "delay_scores",
            {}
        )

        timing_options_parts = []

        for delay in SUPPORTED_DELAYS:

            score_data = timing_scores.get(
                delay,
                {}
            )

            final_timing_score = safe_float(
                score_data.get(
                    "final_score",
                    0.0
                ),
                0.0
            )

            timing_options_parts.append(
                f"{delay} min : "
                f"{final_timing_score:.0%}"
            )

        timing_options = " | ".join(
            timing_options_parts
        )

        selected_delay = int(
            safe_float(
                result.get(
                    "delay_min",
                    SUPPORTED_DELAYS[0]
                    if SUPPORTED_DELAYS
                    else 2
                ),
                2
            )
        )

        if SUPPORTED_DELAYS:

            selected_delay = max(
                min(SUPPORTED_DELAYS),
                min(
                    max(SUPPORTED_DELAYS),
                    selected_delay
                )
            )

        evidence_pct = (
            safe_float(
                result.get(
                    "evidence_confidence",
                    0.0
                ),
                0.0
            ) * 100
        )

        agreement_pct = (
            safe_float(
                result.get(
                    "model_agreement",
                    0.0
                ),
                0.0
            ) * 100
        )

        timing_features = result.get(
            "timing_features",
            {}
        )

        timing_urgency_pct = (
            safe_float(
                timing_features.get(
                    "timing_urgency",
                    0.0
                ),
                0.0
            ) * 100
        )

        observed_timing = result.get(
            "timing_evidence",
            {}
        ).get(
            "observed_scores",
            {}
        )

        observed_options = " | ".join(
            f"{delay}m:"
            f"{safe_float(observed_timing.get(delay, 0.5), 0.5):.0%}"
            for delay in SUPPORTED_DELAYS
        )

        # ------------------------------------------------------------------
        # Pics >10x
        # ------------------------------------------------------------------

        isolated_high_line = ""

        if (
            mode != "ultra"
            and isolated_high
        ):

            isolated_high_line = (
                f"\n🧱 Pics &gt;10x isolés des statistiques normales : "
                f"{len(isolated_high)}"
            )

        # ------------------------------------------------------------------
        # Heure à risque
        # ------------------------------------------------------------------

        high_cote_hour_info = result.get(
            "high_cote_hour",
            {}
        )

        high_hour_obs = int(
            safe_float(
                high_cote_hour_info.get(
                    "observations",
                    0
                ),
                0
            )
        )

        high_hour_alert = ""

        if high_hour_obs >= 1:

            avg_hc = high_cote_hour_info.get(
                "average_cote"
            )

            max_hc = high_cote_hour_info.get(
                "max_cote"
            )

            avg_str = (
                f" | moy {avg_hc}x"
                if avg_hc is not None
                else ""
            )

            max_str = (
                f" | max {max_hc}x"
                if max_hc is not None
                else ""
            )

            high_hour_alert = (
                f"\n🔴 <b>Heure à risque :</b> "
                f"{high_hour_obs} apparition(s) "
                f"de cotes &gt;10x enregistrée(s) "
                f"à {pred_time.hour:02d}h"
                f"{avg_str}"
                f"{max_str}"
                f" — restez vigilant !"
            )

        # ------------------------------------------------------------------
        # Signal temporel >10x
        # ------------------------------------------------------------------

        _ts = result.get(
            "high_cote_time_signal",
            {}
        )

        _ts_prob = safe_float(
            _ts.get(
                "probability",
                0.0
            ),
            0.0
        )

        _ts_count = int(
            safe_float(
                _ts.get(
                    "observations",
                    0
                ),
                0
            )
        )

        _ts_alert = ""

        if (
            _ts_count >= 1
            and _ts_prob >= 0.30
        ):

            _ts_avg = _ts.get(
                "avg_cote"
            )

            _ts_max = _ts.get(
                "max_cote"
            )

            _ts_closest = safe_float(
                _ts.get(
                    "closest_minutes",
                    0
                ),
                0
            )

            _ts_label = _ts.get(
                "label",
                "low"
            )

            _icon = (
                "🔴"
                if _ts_label == "high"
                else "🟡"
            )

            _avg_s = (
                f" | moy {_ts_avg}x"
                if _ts_avg is not None
                else ""
            )

            _max_s = (
                f" | max {_ts_max}x"
                if _ts_max is not None
                else ""
            )

            _win_s = (
                f"±{_ts_closest:.0f} min autour de "
                f"{pred_time.hour:02d}h"
                f"{pred_time.minute:02d}"
            )

            _sources = [
                "observations passées"
            ]

            if result.get(
                "pattern_sequence"
            ):

                _sources.append(
                    "pattern actuel"
                )

            _src_s = " + ".join(
                _sources
            )

            _ts_alert = (
                f"\n{_icon} <b>Probabilité cote &gt;10x "
                f"prochaines minutes : "
                f"{_ts_prob * 100:.0f}%</b>\n"
                f"   📍 {_ts_count} obs. dans "
                f"{_win_s}"
                f"{_avg_s}"
                f"{_max_s}\n"
                f"   🧠 Sources : {_src_s}"
            )

        # ══════════════════════════════════════════════════════════════════
        # MESSAGE FINAL
        # ══════════════════════════════════════════════════════════════════

        target = result.get(
            "target",
            "?"
        )

        assurance = result.get(
            "assurance",
            "?"
        )

        fiabilite = result.get(
            "fiabilite",
            "?"
        )

        target_range = TARGET_RANGES.get(
            mode,
            (None, None)
        )
        range_line = ""

        if target_range[0] is not None:
            range_line = (
                f"📐 <b>Plage du mode :</b> "
                f"{target_range[0]:.0f}x → "
                f"{target_range[1]:.0f}x\n"
            )

        txt = (
            f"🚀 <b>PRÉDICTION {html_mod.escape(str(mode).upper())} v6.0</b>\n"
            f"{header}\n"
            f"{range_line}"
            f"🎯 <b>Cible :</b> {target}x\n"
            f"🛡️ <b>Assurance :</b> {assurance}x\n"
            f"⚡ <b>Fiabilité :</b> {fiabilite}x\n"
            f"{path_line}\n"
            f"📊 <b>Score IA :</b> "
            f"{result.get('score', 0)} "
            f"({result.get('pertinence', 0)}%)\n"
            f"🧪 <b>Confiance des preuves :</b> "
            f"{evidence_pct:.1f}%\n"
            f"🤝 <b>Accord des modèles :</b> "
            f"{agreement_pct:.1f}%\n"
            f"\n"
            f"📈 <b>Analyse :</b>\n"
            f"• Moyenne : "
            f"{result.get('moyenne', 0)}x"
            f"  | Écart : "
            f"{result.get('ecart', 0)}\n"
            f"• Série creuse : "
            f"{result.get('serie_creuse', 0)}\n"
            f"• Stabilité : "
            f"{result.get('stability', 0)}"
            f"  | Entropie : "
            f"{result.get('entropy', 0)}\n"
            f"\n"
            f"🧮 <b>Modèles :</b>\n"
            f"• Markov : "
            f"{result.get('markov', 0)}"
            f"  | Bayes : "
            f"{safe_float(result.get('bayes', 0), 0.0) * 100:.1f}%"
            f"{memory_info}\n"
            f"\n"
            f"⏰ <b>Timing adaptatif :</b>\n"
            f"• Choix IA : "
            f"<b>{selected_delay} minute(s)</b>\n"
            f"• Comparaison : "
            f"<code>{timing_options}</code>\n"
            f"• Urgence temporelle : "
            f"{timing_urgency_pct:.1f}%\n"
            f"• Historique observé : "
            f"<code>{observed_options}</code>\n"
            f"• Fenêtre : "
            f"<code>{result.get('fenetre', ('?', '?'))[0]} "
            f"→ "
            f"{result.get('fenetre', ('?', '?'))[1]}</code>\n"
            f"• Optimal : "
            f"<code>{result.get('optimal_time', '?')}</code>\n"
            f"• Délai prédit : "
            f"{result.get('delay_sec', '?')}s\n"
            f"\n"
            f"{pat_txt}"
            f"{header}\n"
            f"\n"
            f"🔥 <b>Stratégie :</b> "
            f"{result.get('strategy', '?')}"
            f"{mem_line}\n"
            f"{high_hour_alert}"
            f"{_ts_alert}"
            f"{isolated_high_line}"
            f"\n"
            f"<i>Donnez votre feedback après la partie !</i>"
        )

        # ══════════════════════════════════════════════════════════════════
        # 11. ENVOI DU MESSAGE AVANT SAUVEGARDE
        # ══════════════════════════════════════════════════════════════════

        sent = _safe_send_message(
            chat_id,
            txt,
            parse_mode="HTML",
            reply_markup=feedback_keyboard(prediction_id)
        )

        if sent is None:
            logging.warning(
                "[PREDICTION] message Telegram non envoyé"
            )

        # ══════════════════════════════════════════════════════════════════
        # 12. HISTORIQUE
        # ══════════════════════════════════════════════════════════════════

        history.append({
            "chat_id": chat_id,
            "mode": mode,
            "target": target,
            "score": result.get(
                "score",
                final_score
            ),
            "memory_score": memory_result.get(
                "memory_score",
                0.5
            ),
            "memory_confidence": memory_result.get(
                "confidence",
                0.0
            ),
            "memory_samples": memory_result.get(
                "samples",
                0
            ),
            "delay_min": result.get(
                "delay_min"
            ),
            "delay_sec": result.get(
                "delay_sec"
            ),
            "time": pred_time.isoformat(),
        })

        user_history.setdefault(
            chat_id,
            []
        ).append(
            {
                "prediction_id": prediction_id,
                "time": pred_time.isoformat(),
                "mode": mode,
                "target": target,
                "max_cote": None,
                "status": "pending",
                "score": result.get("pertinence", 0),
            }
        )

        # ══════════════════════════════════════════════════════════════════
        # 13. SAUVEGARDE NON BLOQUANTE
        # ══════════════════════════════════════════════════════════════════

        threading.Thread(
            target=save_all,
            daemon=True
        ).start()

    except Exception as e:

        logging.error(
            f"[_handle_mode] {e}",
            exc_info=True
        )

        _safe_send_message(
            chat_id,
            f"❌ <b>Erreur interne :</b>\n"
            f"<code>{html_mod.escape(type(e).__name__)}: "
            f"{html_mod.escape(str(e))}</code>\n\n"
            f"Veuillez réessayer ou envoyer de nouvelles cotes.",
            parse_mode="HTML"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK — MAX COTE
# ══════════════════════════════════════════════════════════════════════════════

def _handle_max_rating_input(
    msg,
    status: str,
    details: dict
):
    """
    Récupère la cote maximale réellement observée.
    """
    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    try:

        raw = str(
            msg.text
            or ""
        ).replace(",", ".").strip()

        max_cote = float(
            raw
        )

        if not (
            1.0 <= max_cote <= 1000.0
        ):
            raise ValueError

    except (
        TypeError,
        ValueError
    ):

        _safe_send_message(
            msg.chat.id,
            "❌ Cote invalide.\n"
            "Exemple : <code>12.5</code>",
            parse_mode="HTML"
        )

        bot.register_next_step_handler(
            msg,
            _handle_max_rating_input,
            status=status,
            details=details
        )

        return

    next_details = dict(
        details or {}
    )

    next_details["max_cote"] = max_cote

    prompt = _safe_send_message(
        msg.chat.id,
        "📉 Combien de parties perdues avec une cote "
        "<b>inférieure à 1.99x</b> y a-t-il eu avant cette issue ?\n\n"
        "Répondez par un nombre entier.\n"
        "Exemple : <code>3</code>",
        parse_mode="HTML"
    )

    if prompt:

        bot.register_next_step_handler(
            prompt,
            _handle_low_cote_losses_input,
            status=status,
            details=next_details
        )


def build_prediction_error_autopsy(feedback_record: dict) -> dict:
    """Diagnose feedback dimensions independently without inventing evidence.

    Directional Gain/Loss, measured elapsed time, and qualitative early/late
    feedback remain separate evidence types. This function is diagnostic only:
    it does not change model weights, targets, or timing decisions.
    """
    record = feedback_record if isinstance(feedback_record, dict) else {}
    prediction = record.get("prediction") or {}
    if not isinstance(prediction, dict):
        prediction = {}

    status = str(record.get("status", "")).lower()
    mode = record.get("mode")
    predicted_target = _safe_float(prediction.get("target"), float("nan"))
    actual_max = _safe_float(record.get("max_cote"), float("nan"))
    target_error = None
    target_gap_pct = None
    target_status = "unavailable"

    if math.isfinite(predicted_target) and predicted_target > 0 and math.isfinite(actual_max):
        target_error = round(actual_max - predicted_target, 4)
        target_gap_pct = round((target_error / predicted_target) * 100.0, 2)
        target_status = "target_reached" if actual_max >= predicted_target else "target_not_reached"

    predicted_delay = _safe_float(
        record.get("predicted_delay_sec", record.get("predicted_delay")),
        float("nan"),
    )
    actual_delay = _safe_float(record.get("actual_delay_sec"), float("nan"))
    delay_source = record.get("actual_delay_source")
    timing_error = None
    timing_status = "not_measured"
    if delay_source == "observed_datetime" and math.isfinite(actual_delay) and math.isfinite(predicted_delay):
        timing_error = round(actual_delay - predicted_delay, 2)
        if abs(timing_error) <= 30.0:
            timing_status = "within_30_seconds"
        elif timing_error > 0:
            timing_status = "observed_later_than_predicted"
        else:
            timing_status = "observed_earlier_than_predicted"
    elif status in ("early", "late") or record.get("timing_adjustment_minutes") in (-1, 1):
        # A qualitative correction indicates direction only, not elapsed time.
        timing_status = "qualitative_feedback_only"

    historical = prediction.get("historical_comparison") or {}
    if not isinstance(historical, dict):
        historical = {}
    all_similar = historical.get("all_similar") or {}
    if not isinstance(all_similar, dict):
        all_similar = {}

    uncertainty = _safe_float(prediction.get("ensemble_uncertainty"), float("nan"))
    novelty = _safe_float(prediction.get("novelty_score"), float("nan"))
    confidence = _safe_float(record.get("decision_confidence"), float("nan"))
    gain_rate = _safe_float(all_similar.get("weighted_gain_rate"), float("nan"))
    similar_count = int(max(0, _safe_float(all_similar.get("samples"), 0)))

    warnings = []
    if math.isfinite(novelty) and novelty >= 0.70:
        warnings.append("high_context_novelty")
    if math.isfinite(uncertainty) and uncertainty >= 0.60:
        warnings.append("high_ensemble_uncertainty")
    if similar_count >= 5 and math.isfinite(gain_rate) and gain_rate < 0.40:
        warnings.append("weak_historical_outcomes_in_similar_contexts")
    if similar_count < 5:
        warnings.append("limited_historical_support")

    direction_label = (
        "gain" if status == "gain" else "loss" if status in ("loss", "perte")
        else "qualitative_or_other"
    )
    return {
        "version": 1,
        "diagnostic_only": True,
        "mode": mode,
        "direction": {
            "feedback": direction_label,
            "target_status": target_status,
            "target_error": target_error,
            "target_gap_percent": target_gap_pct,
        },
        "timing": {
            "evidence_source": delay_source,
            "status": timing_status,
            "predicted_delay_sec": round(predicted_delay, 2) if math.isfinite(predicted_delay) else None,
            "actual_delay_sec": round(actual_delay, 2) if delay_source == "observed_datetime" and math.isfinite(actual_delay) else None,
            "timing_error_sec": timing_error,
            "qualitative_adjustment_minutes": record.get("timing_adjustment_minutes") if timing_status == "qualitative_feedback_only" else None,
        },
        "context": {
            "historical_similar_samples": similar_count,
            "historical_weighted_gain_rate": round(gain_rate, 4) if math.isfinite(gain_rate) else None,
            "ensemble_uncertainty": round(uncertainty, 4) if math.isfinite(uncertainty) else None,
            "novelty_score": round(novelty, 4) if math.isfinite(novelty) else None,
            "decision_confidence": round(confidence, 4) if math.isfinite(confidence) else None,
            "warnings": warnings,
        },
        "interpretation": "Indices descriptifs uniquement ; aucune cause n'est affirmée et aucun moteur n'est réajusté à ce stade.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK FINAL
# ══════════════════════════════════════════════════════════════════════════════

def _timing_feedback_adjustment_minutes(status, explicit_adjustment=None):
    """Return the user's correction direction, not the timing-learning label."""
    try:
        adjustment = float(explicit_adjustment)
    except (TypeError, ValueError):
        adjustment = None

    if adjustment in (-1.0, 1.0):
        return int(adjustment)

    # Internally, "late" means the predicted timing was too late (move it
    # earlier); "early" means it was too early (move it later).
    if status == "late":
        return -1
    if status == "early":
        return 1
    return None


def _finalize_feedback(
    msg,
    status: str,
    details: dict
):
    """
    Finalise un feedback.

    IMPORTANT :
    - gain/perte => apprentissage directionnel ;
    - early/late => apprentissage temporel uniquement ;
      ils ne deviennent jamais une perte directionnelle.
    """

    chat_id = msg.chat.id

    details = details or {}

    pred = details.get(
        "pred"
    )
    prediction_id = str(
        details.get("prediction_id")
        or (pred or {}).get("prediction_id")
        or ""
    )
    direction_status = status if status in ("gain", "perte", "loss") else None
    learning_status = direction_status or status
    timing_adjustment_minutes = _timing_feedback_adjustment_minutes(
        status,
        details.get("timing_adjustment_minutes"),
    )

    try:

        # ══════════════════════════════════════════════════════════════════
        # 1. COTE MAX
        # ══════════════════════════════════════════════════════════════════

        raw_max = details.get(
            "max_cote"
        )

        if raw_max is None:
            raw_max = msg.text

        max_cote = float(
            str(raw_max)
            .replace(",", ".")
            .strip()
        )

        if not (
            1.0 <= max_cote <= 1000.0
        ):
            raise ValueError(
                "cote max hors limites"
            )

        # ══════════════════════════════════════════════════════════════════
        # 2. DONNÉES PRÉDICTION
        # ══════════════════════════════════════════════════════════════════

        result = (
            pred.get(
                "result",
                {}
            )
            if pred
            else {}
        )

        mode = (
            pred.get(
                "mode",
                "?"
            )
            if pred
            else "?"
        )

        pred_iso = (
            pred.get(
                "pred_time"
            )
            if pred
            else None
        )

        real_time_str = details.get(
            "real_time"
        )

        # ══════════════════════════════════════════════════════════════════
        # 3. HEURE RÉELLE / DELAI RÉEL
        # ══════════════════════════════════════════════════════════════════

        delay_sec = None
        actual_dt = None

        if (
            real_time_str
            and pred_iso
        ):

            try:

                pred_dt = datetime.datetime.fromisoformat(
                    pred_iso
                )

                actual_dt = parse_real_datetime(
                    real_time_str,
                    pred_dt
                )

                if actual_dt:

                    delay_sec = (
                        actual_dt - pred_dt
                    ).total_seconds()

                    # Si le parsing a choisi le jour suivant ou précédent
                    # de manière aberrante, on évite une valeur impossible.
                    if abs(delay_sec) > 1800:
                        delay_sec = None
                        actual_dt = None

            except Exception as e:

                logging.warning(
                    f"[DELAY PARSE] {e}"
                )

        # Les boutons -1/+1 sont auto-suffisants : ils décrivent directement
        # l'écart par rapport au délai prédit, sans dépendre d'un autre chat
        # ou d'une prédiction plus récente.
        timing_offset_seconds = None
        if delay_sec is None and details.get("timing_offset_seconds") is not None:
            timing_offset_seconds = safe_float(details.get("timing_offset_seconds"), 0.0)

        # Aucun délai réel n'est fabriqué à partir de -1/+1 min.
        # Ces boutons donnent une indication qualitative (trop tard/trop tôt),
        # pas une mesure exacte de l'heure d'issue.
        timing_learning_delay_sec = delay_sec

        # ══════════════════════════════════════════════════════════════════
        # 4. PERTES <1.99x
        # ══════════════════════════════════════════════════════════════════

        low_cote_losses = max(
            0,
            int(
                safe_float(
                    details.get(
                        "low_cote_losses",
                        0
                    ),
                    0
                )
            )
        )

        # ══════════════════════════════════════════════════════════════════
        # 5. FEEDBACK HISTORY
        # ══════════════════════════════════════════════════════════════════

        feedback_record = {
            "chat_id": chat_id,

            "status": status,

            "learning_status": learning_status,
            "feedback_action": details.get("feedback_action"),
            "timing_adjustment_minutes": timing_adjustment_minutes,

            "max_cote": max_cote,

            "low_cote_losses": low_cote_losses,

            "real_time": real_time_str,
            "real_time_window": details.get("real_time_window"),

            "delay_sec": result.get("delay_sec"),
            "predicted_delay_sec": result.get("delay_sec"),
            "actual_delay_sec": delay_sec,
            "timing_learning_delay_sec": timing_learning_delay_sec,
            "timing_offset_seconds": timing_offset_seconds,
            # Provenance explicite : +/-1 min est une indication qualitative,
            # jamais une mesure réelle du délai écoulé.
            "actual_delay_source": (
                "observed_datetime" if delay_sec is not None
                else "timing_feedback_inferred" if timing_offset_seconds is not None
                else None
            ),

            "pred_time": pred_iso,

            "predicted_delay": result.get(
                "delay_min"
            ),

            "predicted_delay_sec": result.get(
                "delay_sec"
            ),

            "delay_min": result.get(
                "delay_min"
            ),

            "mode": mode,

            "cotes": (
                list(
                    pred.get(
                        "cotes",
                        []
                    )
                )
                if pred
                else []
            ),
            "pattern_sequence": (
                list((result or {}).get("pattern_sequence") or pred.get("cotes", []))
                if pred else []
            ),
            "sequence_features": extract_sequence_features(
                (result or {}).get("pattern_sequence") or (pred or {}).get("cotes", [])
            ),

            "experience_id": (
                pred.get(
                    "experience_id"
                )
                if pred
                else None
            ),

            "prediction_id": prediction_id or None,

            "timing_context_key": (
                result.get(
                    "timing_context_key"
                )
                if pred
                else None
            ),

            "timing_features": (
                result.get(
                    "timing_features",
                    {}
                )
                if pred
                else {}
            ),

            "timing_evidence": (
                result.get(
                    "timing_evidence",
                    {}
                )
                if pred
                else {}
            ),

            "delay_scores": (
                result.get(
                    "delay_scores",
                    {}
                )
                if pred
                else {}
            ),

            "prediction": result,
            "ensemble_signals": result.get("ensemble_signals", {}),
            "ensemble_weights": result.get("ensemble_weights", {}),
            "ensemble_score": result.get("ensemble_score"),
            "ensemble_uncertainty": result.get("ensemble_uncertainty"),
            "ensemble_feedback_samples": result.get("ensemble_feedback_samples", 0),

            "decision": result.get("decision"),
            "decision_confidence": result.get("decision_confidence"),
            "direction_probability": result.get("direction_probability"),
            "timing_interval": result.get("timing_interval"),
            "novelty_score": result.get("novelty_score"),

            "time": datetime.datetime.now().isoformat(),

            # Informations importantes pour l'apprentissage temporel
            "actual_datetime": (
                actual_dt.isoformat()
                if actual_dt
                else None
            ),

            "predicted_datetime": pred_iso,
        }

        # Autopsie séparée des erreurs : Gain/Perte reste directionnel,
        # et seul un délai issu d'une heure réelle est traité comme mesure temporelle.
        try:
            feedback_record["error_autopsy"] = build_prediction_error_autopsy(feedback_record)
        except Exception as autopsy_exc:
            logging.warning("[ERROR AUTOPSY] %s", autopsy_exc)
            feedback_record["error_autopsy"] = {
                "version": 1,
                "diagnostic_only": True,
                "status": "error",
            }

        feedback_history.append(
            feedback_record
        )

        # ══════════════════════════════════════════════════════════════════
        # 6. FEEDBACK SAMPLE
        # ══════════════════════════════════════════════════════════════════

        sample = create_feedback_sample(
            chat_id=chat_id,
            prediction=pred or {},
            status=learning_status,
            actual_datetime=actual_dt,
            max_cote_reached=max_cote,
            low_cote_losses=low_cote_losses,
        )
        if isinstance(sample, dict):
            sample["feedback_action"] = details.get("feedback_action")
            sample["timing_adjustment_minutes"] = timing_adjustment_minutes

        register_feedback_sample(
            sample
        )

        # ══════════════════════════════════════════════════════════════════
        # 7. DATASET — UNIQUEMENT GAIN/PERTE
        # ══════════════════════════════════════════════════════════════════

        definitive_result = direction_status in ("gain", "perte", "loss")

        if definitive_result:

            ts = build_training_sample_from_feedback(
                sample
            )

            if ts:
                training_dataset.append(
                    ts
                )

        # ══════════════════════════════════════════════════════════════════
        # 8. MÉMOIRE D'EXPÉRIENCE
        # ══════════════════════════════════════════════════════════════════

        _safe_send_message(
            chat_id,
            "🧠 <b>Mise à jour de la mémoire IA...</b>",
            parse_mode="HTML"
        )

        if definitive_result:
            update_experience_memory(
                prediction=pred or {},
                status=learning_status,
                real_time=real_time_str,
                real_max_cote=max_cote,
                delay_seconds=delay_sec,
                low_cote_losses=low_cote_losses,
            )
        # early/late is intentionally excluded from experience_memory's
        # directional reliability/target-error updates; dedicated timing
        # memories above record it independently.

        # ══════════════════════════════════════════════════════════════════
        # 9. META DECISION ENGINE
        # ══════════════════════════════════════════════════════════════════

        pred_model_outputs = (
            pred.get(
                "model_outputs",
                {}
            )
            if pred
            else {}
        )

        pred_meta_context = (
            pred.get(
                "meta_context",
                "CALM"
            )
            if pred
            else "CALM"
        )

        if (
            pred_model_outputs
            and definitive_result
        ):

            update_meta_model_stats(
                pred_meta_context,
                pred_model_outputs,
                learning_status == "gain",
                safe_float(
                    result.get(
                        "score",
                        0.50
                    ),
                    0.50
                ),
            )

        # ══════════════════════════════════════════════════════════════════
        # 11. PATTERN MEMORY
        # ══════════════════════════════════════════════════════════════════

        if (
            pred
            and definitive_result
        ):

            pattern_sequence = (
                pred.get(
                    "result",
                    {}
                ).get(
                    "pattern_sequence",
                    []
                )
            )

            update_pattern_memory(
                pattern_sequence,
                learning_status
            )

        # ==============================================================
        # PATTERN → DÉLAI → RÉSULTAT — v6.0
        # ==============================================================
        if pred:
            pattern_sequence = (
                pred.get(
                    "result",
                    {}
                ).get(
                    "pattern_sequence",
                    []
                )
            )

            predicted_delay = _safe_float(
                result.get("delay_min"),
                0.0
            )

            if predicted_delay <= 0:
                predicted_delay = (
                    _safe_float(
                        result.get("delay_sec"),
                        0.0
                    ) / 60.0
                )

            if predicted_delay > 0:
                update_pattern_timing_memory(
                    pattern_seq=pattern_sequence,
                    mode=mode,
                    predicted_delay=int(round(predicted_delay)),
                    actual_delay=delay_sec,
                    status=status,
                    confidence=_safe_float(result.get("timing_confidence"), 0.50),
                    actual_delay_source=("observed_datetime" if delay_sec is not None else None),
                )

        # ══════════════════════════════════════════════════════════════════
        # 12. CONTEXT MEMORY
        # ══════════════════════════════════════════════════════════════════

        if (
            pred
            and definitive_result
        ):

            cotes_for_ctx = (
                pred.get(
                    "cotes",
                    []
                )
            )

            if cotes_for_ctx:

                ctx = detect_market_context(
                    cotes_for_ctx
                )

                pat_key = build_pattern_key(
                    cotes_for_ctx
                )

                success = (
                    learning_status == "gain"
                )

                confidence = (
                    safe_float(
                        result.get(
                            "pertinence",
                            50.0
                        ),
                        50.0
                    ) / 100.0
                )

                update_context_memory(
                    ctx,
                    pat_key,
                    success,
                    confidence
                )

                # ----------------------------------------------------------
                # Per-model performance
                # ----------------------------------------------------------

                try:

                    # Utiliser les sorties capturées au moment de la prédiction.
                    # Ne jamais recalculer les modèles après avoir reçu le feedback.
                    prob_sources = {
                        name: safe_float(output.get("probability", 0.50), 0.50)
                        for name, output in (pred_model_outputs or {}).items()
                        if isinstance(output, dict)
                    }

                    for model_name, prob in prob_sources.items():

                        prob = safe_float(prob, 0.50)

                        model_correct = (
                            (prob >= 0.50)
                            == success
                        )

                        update_model_performance(
                            ctx,
                            model_name,
                            model_correct
                        )

                except Exception as e:

                    logging.warning(
                        f"[MODEL PERFORMANCE] {e}"
                    )

        # ══════════════════════════════════════════════════════════════════
        # 13. TIMING MEMORY
        # ══════════════════════════════════════════════════════════════════

        if pred:

            update_timing_feedback(
                pred,
                status,
                actual_delay=(delay_sec if delay_sec is not None else None)
            )
            update_sequence_feature_memory(
                sequence=(result or {}).get("pattern_sequence") or (pred or {}).get("cotes", []),
                mode=mode,
                status=status,
                predicted_delay=result.get("delay_min", predicted_delay if "predicted_delay" in locals() else None),
                actual_delay=delay_sec,
                actual_delay_source=("observed_datetime" if delay_sec is not None else None),
                timing_offset_seconds=timing_offset_seconds,
                confidence=_safe_float(result.get("timing_confidence"), 0.50),
            )

            # La calibration V6 doit voir immédiatement le nouveau feedback.
            try:
                refresh_v6_timing_calibration()
            except Exception as e:
                logging.warning(f"[V6 TIMING REFRESH] {e}")

        # ══════════════════════════════════════════════════════════════════
        # 14. RETRAINING AUTOMATIQUE
        # ══════════════════════════════════════════════════════════════════

        directional_feedback_count = sum(
            1 for item in feedback_history
            if isinstance(item, dict)
            and item.get("status", item.get("learning_status")) in ("gain", "perte", "loss")
        )

        if (
            definitive_result
            and directional_feedback_count >= 60
            and directional_feedback_count % 60 == 0
        ):
            logging.info(
                "[AUTO RETRAIN] seuil atteint : %s feedbacks directionnels; le moteur temporel reste indépendant.",
                directional_feedback_count,
            )
            threading.Thread(
                target=_train_model_on_dataset,
                args=(list(training_dataset),),
                daemon=True,
                name=f"retrain-{directional_feedback_count}",
            ).start()

        # ══════════════════════════════════════════════════════════════════
        # 15. QUALITÉ
        # ══════════════════════════════════════════════════════════════════

        quality = compute_prediction_quality(
            sample
        )

        # L'historique doit refléter le résultat final, et non rester bloqué
        # sur l'état "prédiction envoyée".
        for entry in user_history.get(chat_id, []):
            if not isinstance(entry, dict):
                continue
            same_prediction = (
                prediction_id
                and str(entry.get("prediction_id", ""))
                == prediction_id
            )
            same_time = (
                not prediction_id
                and pred_iso
                and entry.get("time") == pred_iso
            )
            if same_prediction or same_time:
                entry["status"] = status
                entry["learning_status"] = learning_status
                entry["max_cote"] = max_cote
                entry["real_time"] = real_time_str
                entry["feedback_action"] = details.get("feedback_action")
                entry["timing_adjustment_minutes"] = timing_adjustment_minutes
                break

        target_value = safe_float(
            result.get(
                "target",
                0
            ),
            0.0
        )

        if target_value > 0:

            precision = (
                max_cote
                / target_value
            ) * 100.0

        else:

            precision = 0.0

        # On évite un affichage aberrant.
        precision = max(
            0.0,
            min(
                999.0,
                precision
            )
        )

        result_icon = (
            "✅"
            if status == "gain"
            else
            "❌"
            if status in (
                "perte",
                "loss"
            )
            else
            "⏱"
        )

        result_text = {
            "gain": "GAIN",
            "perte": "PERTE",
            "loss": "PERTE",
        }.get(
            status,
            (
                f"{timing_adjustment_minutes:+d} MIN"
                if timing_adjustment_minutes in (-1, 1)
                else status.upper()
            )
        )

        total_fb = len(
            feedback_history
        )

        directional_feedback = sum(
            1
            for f in feedback_history
            if isinstance(f, dict)
            and str(f.get("status") or f.get("learning_status") or "").lower()
                in ("gain", "perte", "loss")
        )

        timing_correction_feedback = sum(
            1
            for f in feedback_history
            if isinstance(f, dict)
            and str(f.get("status") or f.get("learning_status") or "").lower()
                in ("early", "late")
        )

        wins = sum(
            1
            for f in feedback_history
            if isinstance(f, dict)
            and str(f.get("status") or f.get("learning_status") or "").lower()
                == "gain"
        )

        rate = (
            wins
            / max(
                directional_feedback,
                1
            )
        ) * 100

        if definitive_result:

            feedback_kind = (
                "Résultat directionnel enregistré "
                "et utilisable pour l'apprentissage."
            )

        else:

            feedback_kind = (
                "Correction de timing enregistrée "
                "sans modifier le taux de gain."
            )

        # ══════════════════════════════════════════════════════════════════
        # 16. CONFIRMATION
        # ══════════════════════════════════════════════════════════════════

        txt = (
            f"{result_icon} <b>FEEDBACK ENREGISTRÉ</b>\n\n"

            f"📊 <b>Résultat :</b> "
            f"{html_mod.escape(result_text)}\n"

            f"🧭 <b>Type :</b> "
            f"{feedback_kind}\n"

            f"🎯 <b>Cote max :</b> "
            f"{max_cote:.3f}x\n"

            f"📉 <b>Pertes ≤ 1.99x avant l'issue :</b> "
            f"{low_cote_losses}\n"

            f"📈 <b>Cible prédite :</b> "
            f"{target_value if target_value else '?'}x\n"

            f"🎲 <b>Précision :</b> "
            f"{precision:.1f}%\n"

            f"⭐ <b>Qualité prédiction :</b> "
            f"{safe_float(quality, 0.0) * 100:.1f}%\n"
        )

        if delay_sec is not None:

            predicted_delay = safe_float(
                result.get(
                    "delay_sec",
                    0
                ),
                0
            )

            txt += (
                f"\n⏱ <b>Timing :</b>\n"
                f"• Délai prédit : "
                f"{predicted_delay:.0f}s\n"
                f"• Délai réel : "
                f"{delay_sec:.0f}s\n"
                f"• Écart : "
                f"{delay_sec - predicted_delay:+.0f}s\n"
            )

        txt += (
            f"\n🧠 <b>IA mise à jour !</b>\n"
            f"• Feedbacks totaux : "
            f"{total_fb} "
            f"({directional_feedback} directionnels | "
            f"{timing_correction_feedback} corrections timing)\n"
            f"• Taux réussite : "
            f"{rate:.1f}%\n"
            f"• Expériences mémorisées : "
            f"{len(experience_memory)}\n"
            f"• Échantillons d'apprentissage : "
            f"{len(training_dataset)}\n\n"
            f"<i>La mémoire temporelle et directionnelle "
            f"évoluent séparément.</i>"
        )

        _safe_send_message(
            chat_id,
            txt,
            parse_mode="HTML"
        )

        # ══════════════════════════════════════════════════════════════════
        # 17. NETTOYAGE
        # ══════════════════════════════════════════════════════════════════

        bucket = pending_predictions.get(chat_id)
        if isinstance(bucket, dict) and "result" not in bucket:
            if prediction_id:
                bucket.pop(prediction_id, None)
            if not bucket:
                pending_predictions.pop(chat_id, None)
        else:
            pending_predictions.pop(chat_id, None)

        _prune_memory()

        save_all()

        logging.info(
            "[FEEDBACK] status=%s max=%.3fx delay=%s quality=%.3f",
            status,
            max_cote,
            delay_sec,
            safe_float(quality, 0.0)
        )

    except ValueError:

        _safe_send_message(
            chat_id,
            "❌ Valeur invalide."
        )

    except Exception as e:

        _safe_send_message(
            chat_id,
            f"❌ <b>Erreur :</b> "
            f"<code>{html_mod.escape(str(e))}</code>",
            parse_mode="HTML"
        )

        logging.error(
            f"[FEEDBACK ERROR] {e}",
            exc_info=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — START
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    commands=["start"]
)
def cmd_start(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    _safe_send_message(
        msg.chat.id,

        "🚀 <b>Rocket Queen V7</b>\n\n"

        "Envoyez une capture ou des cotes en texte, "
        "puis choisissez votre mode.\n\n"

        "<b>Nouveautés :</b>\n"
        "• 🧠 Mémoire intelligente des expériences\n"
        "• 📚 Apprentissage continu par feedback\n"
        "• ⏱️ Apprentissage séparé du timing\n"
        "• 🔎 Mémoire des patterns gain/perte\n"
        "• 📊 Analyse des observations &gt;10x\n"
        "• 🤖 Réentraînement du modèle",

        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text in [
            "🎯 2X-10X",
            "🚀 Grosse Cote",
            "🔮 Ultra Grosse Cote"
        ]
)
def handle_prediction(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    mode_map = {
        "🎯 2X-10X": "2X-10X",
        "🚀 Grosse Cote": "grosse",
        "🔮 Ultra Grosse Cote": "ultra",
    }

    mode = mode_map.get(
        msg.text
    )

    if not mode:
        return

    _handle_mode(
        msg,
        mode
    )


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION PASSIVE >10X
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text == "📝 Observer >10x"
)
def handle_high_cote_observation_start(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    prompt = _safe_send_message(
        msg.chat.id,

        "📝 <b>Observation passive &gt;10x</b>\n\n"
        "Entrez la cote observée "
        "(strictement supérieure à 10.00x).\n\n"
        "Exemple : <code>18.40</code>",

        parse_mode="HTML"
    )

    if prompt:

        bot.register_next_step_handler(
            prompt,
            _handle_high_cote_value_input
        )


def _handle_high_cote_value_input(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    try:

        cote = float(
            str(msg.text or "")
            .replace(",", ".")
            .replace("x", "")
            .replace("X", "")
            .strip()
        )

        if (
            cote <= 10.0
            or cote > 1000.0
        ):
            raise ValueError

    except (
        TypeError,
        ValueError
    ):

        _safe_send_message(
            msg.chat.id,
            "❌ Cote invalide.\n"
            "Entrez une valeur strictement supérieure à 10.00x."
        )

        bot.register_next_step_handler(
            msg,
            _handle_high_cote_value_input
        )

        return

    prompt = _safe_send_message(
        msg.chat.id,

        f"✅ Cote <code>{cote:.3f}x</code> reçue.\n\n"
        f"À quelle heure exacte a-t-elle été observée ?\n"
        f"(<code>HH:MM</code> ou <code>HH:MM:SS</code>)",

        parse_mode="HTML"
    )

    if prompt:

        bot.register_next_step_handler(
            prompt,
            _handle_high_cote_time_input,
            cote=cote
        )


def _handle_high_cote_time_input(
    msg,
    cote: float
):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    raw = str(
        msg.text or ""
    ).strip()

    parsed = None

    for fmt in (
        "%H:%M:%S",
        "%H:%M"
    ):

        try:

            parsed = datetime.datetime.strptime(
                raw,
                fmt
            )

            break

        except ValueError:
            continue

    if parsed is None:

        _safe_send_message(
            msg.chat.id,

            "❌ Heure invalide.\n"
            "Utilisez <code>HH:MM</code> "
            "ou <code>HH:MM:SS</code>.",

            parse_mode="HTML"
        )

        bot.register_next_step_handler(
            msg,
            _handle_high_cote_time_input,
            cote=cote
        )

        return

    now = datetime.datetime.now()

    observation = {
        "chat_id": msg.chat.id,

        "cote": round(
            float(cote),
            3
        ),

        "time": raw,

        "hour": parsed.hour,

        "minute": parsed.minute,

        "second": (
            parsed.second
            if len(
                raw.split(":")
            ) == 3
            else 0
        ),

        "recorded_at": now.isoformat(),

        "source":
            "manual_high_cote_observation",
    }

    high_cote_observations.append(
        observation
    )

    _prune_memory()

    save_all()

    signal = get_high_cote_hour_signal(
        parsed.hour
    )

    _safe_send_message(
        msg.chat.id,

        f"✅ <b>Observation enregistrée</b>\n"
        f"• Cote : <code>{cote:.3f}x</code>\n"
        f"• Heure : <code>{raw}</code>\n"
        f"• Observations &gt;10x à "
        f"{parsed.hour:02d}h : "
        f"<b>{signal.get('observations', 0)}</b>\n\n"
        f"Cette observation ne lance aucune prédiction "
        f"et servira de mémoire horaire.",

        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE HEURES >10X
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text == "📋 Heures >10x"
)
def handle_show_high_cote_hours(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    chat_id = msg.chat.id

    if not high_cote_observations:

        _safe_send_message(
            chat_id,

            "📋 <b>Aucune observation &gt;10x "
            "enregistrée pour l'instant.</b>\n\n"
            "Utilisez <b>📝 Observer &gt;10x</b> "
            "pour en ajouter.",

            parse_mode="HTML",
            reply_markup=main_keyboard()
        )

        return

    from collections import defaultdict

    groups = defaultdict(list)

    for obs in high_cote_observations:

        try:
            hour = int(
                obs.get(
                    "hour",
                    0
                )
            )

            groups[hour].append(
                obs
            )

        except Exception:
            continue

    lines = [
        "📋 <b>Récapitulatif des heures &gt;10x</b>\n"
    ]

    for hour in sorted(
        groups.keys()
    ):

        obs_list = groups[
            hour
        ]

        cotes = [
            safe_float(
                o.get(
                    "cote",
                    0
                ),
                0.0
            )
            for o in obs_list
        ]

        cotes = [
            c for c in cotes
            if c > 10.0
        ]

        if not cotes:
            continue

        avg_c = (
            sum(cotes)
            / len(cotes)
        )

        max_c = max(
            cotes
        )

        minutes_str = ", ".join(
            f"{int(o.get('minute', 0)):02d}min"
            for o in sorted(
                obs_list,
                key=lambda x:
                    int(
                        x.get(
                            "minute",
                            0
                        )
                    )
            )
        )

        signal = get_high_cote_hour_signal(
            hour
        )

        pressure = safe_float(
            signal.get(
                "pressure",
                0.0
            ),
            0.0
        )

        pressure_icon = (
            "🔴"
            if pressure > 0.6
            else
            "🟡"
            if pressure > 0.3
            else
            "🟢"
        )

        lines.append(
            f"{pressure_icon} <b>{hour:02d}h</b> — "
            f"{len(cotes)} obs. "
            f"| moy <code>{avg_c:.1f}x</code> "
            f"| max <code>{max_c:.1f}x</code>\n"
            f"   ⏱ Minutes : {minutes_str}\n"
        )

    lines.append(
        f"\n<i>Total : "
        f"{len(high_cote_observations)} "
        f"observation(s) mémorisée(s).</i>"
    )

    _safe_send_message(
        chat_id,
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=main_keyboard()
    )


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

@bot.callback_query_handler(
    func=lambda call:
        call.data.startswith("fb_")
)
def handle_feedback_buttons(call):

    chat_id = call.message.chat.id

    raw_action = str(call.data or "")
    action, separator, prediction_id = raw_action.partition(":")

    # Les anciens boutons sans identifiant restent lisibles, mais utilisent
    # uniquement la dernière prédiction du même chat.
    pred = _pending_prediction(
        chat_id,
        prediction_id if separator else None
    )

    if not pred:

        try:
            bot.answer_callback_query(
                call.id,
                "❌ Aucune prédiction en attente"
            )
        except Exception:
            pass

        return

    if pred.get("feedback_in_progress"):
        try:
            bot.answer_callback_query(
                call.id,
                "⏳ Feedback déjà commencé pour cette prédiction."
            )
        except Exception:
            pass
        return

    status_map = {
        "fb_gain": "gain",
        "fb_perte": "perte",
        # -1 min asks to move earlier: the current timing was too late.
        "fb_early": "late",
        # +1 min asks to move later: the current timing was too early.
        "fb_late": "early",
    }

    status = status_map.get(
        action
    )

    if not status:
        return

    if separator and (
        not pred
        or str(pred.get("prediction_id")) != str(prediction_id)
    ):
        try:
            bot.answer_callback_query(
                call.id,
                "❌ Ce bouton appartient à une ancienne prédiction."
            )
        except Exception:
            pass
        return

    pred["feedback_in_progress"] = True
    pred["feedback_status"] = status

    label_map = {
        "fb_gain": "✅ Gain",
        "fb_perte": "❌ Perte",
        "fb_early": "⏪ -1 min",
        "fb_late": "⏩ +1 min",
    }

    prompt_map = {
        "fb_gain":
            "✅ <b>GAIN !</b>",

        "fb_perte":
            "❌ <b>PERTE.</b>",

        "fb_early":
            "⏪ <b>Timing proposé trop tard : correction −1 minute.</b>",

        "fb_late":
            "⏩ <b>Timing proposé trop tôt : correction +1 minute.</b>",
    }

    prompt = prompt_map.get(
        action,
        ""
    )

    # Réponse callback AVANT tout traitement.
    try:

        bot.answer_callback_query(
            call.id,
            label_map.get(
                action,
                ""
            )
        )

    except Exception as e:

        logging.warning(
            f"[CALLBACK] {e}"
        )

    if status in ("early", "late"):
        offset_seconds = -60 if action == "fb_early" else 60

        # Afficher la fenêtre prévue et une fenêtre indicative décalée de 60 s.
        # L'utilisateur saisit ensuite l'intervalle réellement observé.
        pred_result = pred.get("result", {}) if isinstance(pred.get("result"), dict) else {}
        predicted_window = pred_result.get("fenetre", ("?", "?"))
        try:
            win_start = datetime.datetime.strptime(str(predicted_window[0]), "%H:%M:%S")
            win_end = datetime.datetime.strptime(str(predicted_window[1]), "%H:%M:%S")
            shifted_start = (win_start + datetime.timedelta(seconds=offset_seconds)).strftime("%H:%M:%S")
            shifted_end = (win_end + datetime.timedelta(seconds=offset_seconds)).strftime("%H:%M:%S")
            predicted_window_text = f"{predicted_window[0]}–{predicted_window[1]}"
            shifted_window_text = f"{shifted_start}–{shifted_end}"
        except Exception:
            predicted_window_text = "fenêtre affichée dans la prédiction"
            shifted_window_text = "fenêtre décalée de 60 secondes"

        m = _safe_send_message(
            chat_id,
            f"{prompt}\n\n"
            f"⏱ <b>Fenêtre prédite :</b> <code>{html_mod.escape(predicted_window_text)}</code>\n"
            f"➡️ <b>Fenêtre indicative après correction :</b> <code>{html_mod.escape(shifted_window_text)}</code>\n\n"
            f"📌 Saisis la <b>fenêtre réellement observée</b> au format "
            f"<code>HH:MM:SS-HH:MM:SS</code>.\n"
            f"<i>Exemple : 14:54:33-14:55:33</i>\n\n"
            f"Si tu ne connais pas l'intervalle exact, réponds <code>inconnue</code>.",
            parse_mode="HTML"
        )
        details = {
            "pred": pred,
            "prediction_id": pred.get("prediction_id"),
            "real_time": None,
            "real_time_window": None,
            "timing_offset_seconds": offset_seconds,
            "feedback_action": action,
            "timing_adjustment_minutes": int(offset_seconds / 60),
        }
        if m:
            bot.register_next_step_handler(
                m,
                _handle_timing_window_input,
                status=status,
                details=details
            )
        return

    m = _safe_send_message(
        chat_id,
        f"{prompt}\n\n"
        f"📌 À quelle heure a-t-on vu cette cote ? "
        f"(<code>HH:MM</code> ou "
        f"<code>HH:MM:SS</code>)\n\n"
        f"<i>Exemple : 14:32 ou 14:32:45</i>",
        parse_mode="HTML"
    )

    if m:

        bot.register_next_step_handler(
            m,
            _handle_real_time_input,
            status=status,
            pred=pred,
            prediction_id=pred.get("prediction_id")
        )


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK — HEURE RÉELLE
# ══════════════════════════════════════════════════════════════════════════════

def _handle_real_time_input(
    msg,
    status: str,
    pred: dict,
    prediction_id: str = None
):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    chat_id = msg.chat.id

    raw = str(
        msg.text or ""
    ).strip()

    pred_iso = pred.get(
        "pred_time"
    )

    real_time_parsed = None

    if pred_iso:

        try:

            pred_dt = datetime.datetime.fromisoformat(
                pred_iso
            )

            real_time_parsed = parse_real_datetime(
                raw,
                pred_dt
            )

        except Exception as e:

            logging.warning(
                f"[REAL TIME] {e}"
            )

    if real_time_parsed is None:
        if raw.lower() in {"skip", "inconnue", "inconnu", "ignorer", "je ne sais pas"}:
            m2 = _safe_send_message(
                chat_id,
                "ℹ️ Heure réelle non enregistrée. Le feedback directionnel sera conservé, "
                "mais il ne servira pas à confirmer un délai.\n\n"
                "📌 Cote maximum atteinte ? (ex. <code>12.5</code>)",
                parse_mode="HTML"
            )
            if m2:
                bot.register_next_step_handler(
                    m2, _handle_max_rating_input, status=status,
                    details={"pred": pred, "prediction_id": prediction_id or pred.get("prediction_id"), "real_time": None}
                )
            return

        m_retry = _safe_send_message(
            chat_id,
            "⚠️ Heure ambiguë ou hors de la fenêtre de 30 minutes.\n\n"
            "Saisissez l'heure exacte <code>HH:MM:SS</code> (ex. <code>22:14:25</code>), "
            "ou une date complète <code>AAAA-MM-JJ HH:MM:SS</code>. "
            "N'indiquez pas une heure antérieure sans date : le bot ne supposera pas un changement de jour.\n\n"
            "Si l'heure est inconnue, répondez <code>inconnue</code>.",
            parse_mode="HTML"
        )
        if m_retry:
            bot.register_next_step_handler(
                m_retry, _handle_real_time_input, status=status, pred=pred,
                prediction_id=prediction_id or pred.get("prediction_id")
            )
        return

    m2 = _safe_send_message(
        chat_id,

        f"⏱ Heure enregistrée : "
        f"<code>{html_mod.escape(raw)}</code>\n\n"
        f"📌 Quelle était la "
        f"<b>cote maximum</b> atteinte ?\n\n"
        f"<i>Exemple : 12.5</i>",

        parse_mode="HTML"
    )

    if m2:

        bot.register_next_step_handler(
            m2,
            _handle_max_rating_input,
            status=status,
            details={
                "pred": pred,
                "prediction_id": prediction_id or pred.get(
                    "prediction_id"
                ),
                "real_time": raw,
            }
        )


def _handle_timing_window_input(
    msg,
    status: str,
    details: dict
):
    """Capture la fenêtre réellement observée après un feedback -1/+1 minute."""
    if is_duplicate_message(msg.chat.id, msg.text):
        return

    raw = str(msg.text or "").strip()
    chat_id = msg.chat.id
    details = dict(details or {})
    pred = details.get("pred") or {}

    if raw.lower() in {"inconnue", "inconnu", "skip", "ignorer", "je ne sais pas"}:
        details["real_time"] = None
        details["real_time_window"] = None
        m2 = _safe_send_message(
            chat_id,
            "ℹ️ Fenêtre réelle non renseignée. Le feedback de correction sera conservé "
            "comme signal temporel, sans heure réelle mesurée.\n\n"
            "📌 Quelle cote maximum a été atteinte ?\n"
            "<i>Exemple : 12.5</i>",
            parse_mode="HTML"
        )
        if m2:
            bot.register_next_step_handler(
                m2, _handle_max_rating_input, status=status, details=details
            )
        return

    match = re.fullmatch(
        r"\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*",
        raw
    )
    if not match:
        retry = _safe_send_message(
            chat_id,
            "⚠️ Format non reconnu. Saisis les deux heures séparées par un tiret, "
            "par exemple <code>14:54:33-14:55:33</code>.\n"
            "Si tu ne connais pas la fenêtre réelle, réponds <code>inconnue</code>.",
            parse_mode="HTML"
        )
        if retry:
            bot.register_next_step_handler(
                retry, _handle_timing_window_input, status=status, details=details
            )
        return

    def _parse_clock(value):
        fmt = "%H:%M:%S" if value.count(":") == 2 else "%H:%M"
        return datetime.datetime.strptime(value, fmt)

    try:
        start_clock = _parse_clock(match.group(1))
        end_clock = _parse_clock(match.group(2))
        pred_iso = pred.get("pred_time")
        pred_dt = datetime.datetime.fromisoformat(pred_iso) if pred_iso else datetime.datetime.now()

        start_dt = pred_dt.replace(
            hour=start_clock.hour, minute=start_clock.minute, second=start_clock.second, microsecond=0
        )
        end_dt = pred_dt.replace(
            hour=end_clock.hour, minute=end_clock.minute, second=end_clock.second, microsecond=0
        )
        if (end_dt - start_dt).total_seconds() <= 0:
            end_dt += datetime.timedelta(days=1)
        while start_dt - pred_dt > datetime.timedelta(hours=12):
            start_dt -= datetime.timedelta(days=1)
            end_dt -= datetime.timedelta(days=1)
        while pred_dt - end_dt > datetime.timedelta(hours=12):
            start_dt += datetime.timedelta(days=1)
            end_dt += datetime.timedelta(days=1)
        if (end_dt - start_dt).total_seconds() > 300:
            raise ValueError("intervalle trop long")
        midpoint = start_dt + (end_dt - start_dt) / 2
    except Exception:
        retry = _safe_send_message(
            chat_id,
            "⚠️ Les heures sont invalides ou l'intervalle dépasse 5 minutes. "
            "Réessaie, par exemple <code>14:54:33-14:55:33</code>.",
            parse_mode="HTML"
        )
        if retry:
            bot.register_next_step_handler(
                retry, _handle_timing_window_input, status=status, details=details
            )
        return

    details["real_time"] = midpoint.strftime("%Y-%m-%d %H:%M:%S")
    details["real_time_window"] = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "input": raw,
        "source": "user_observed_window",
    }

    m2 = _safe_send_message(
        chat_id,
        f"✅ Fenêtre réellement observée enregistrée : <code>{html_mod.escape(raw)}</code>\n"
        f"⏱ Heure centrale utilisée pour le calcul : <code>{midpoint.strftime('%H:%M:%S')}</code>\n\n"
        f"📌 Quelle cote maximum a été atteinte ?\n"
        f"<i>Exemple : 12.5</i>",
        parse_mode="HTML"
    )
    if m2:
        bot.register_next_step_handler(
            m2, _handle_max_rating_input, status=status, details=details
        )


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK — PERTES <1.99X
# ══════════════════════════════════════════════════════════════════════════════

def _handle_low_cote_losses_input(
    msg,
    status: str,
    details: dict
):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    try:

        raw = str(
            msg.text or ""
        ).strip()

        losses = int(
            raw
        )

        if not (
            0 <= losses <= 10000
        ):
            raise ValueError

    except (
        TypeError,
        ValueError
    ):

        _safe_send_message(
            msg.chat.id,

            "❌ Entrez un nombre entier "
            "entre 0 et 10000.\n"
            "Exemple : <code>3</code>",

            parse_mode="HTML"
        )

        bot.register_next_step_handler(
            msg,
            _handle_low_cote_losses_input,
            status=status,
            details=details
        )

        return

    next_details = dict(
        details or {}
    )

    next_details[
        "low_cote_losses"
    ] = losses

    _finalize_feedback(
        msg,
        status=status,
        details=next_details
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text == "📜 Historique"
)
def handle_history(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    hist = user_history.get(
        msg.chat.id,
        []
    )

    if not hist:

        _safe_send_message(
            msg.chat.id,
            "📭 <b>Historique vide</b>",
            parse_mode="HTML"
        )

        return

    status_labels = {
        "gain": "✅",
        "perte": "❌",
        "loss": "❌",
        "pending": "⏳",
    }
    rendered = []
    for entry in hist[-20:]:
        if isinstance(entry, str):
            rendered.append(entry)
            continue

        if not isinstance(entry, dict):
            continue

        raw_time = entry.get("time") or entry.get("prediction_time") or "?"
        try:
            display_time = datetime.datetime.fromisoformat(
                str(raw_time)
            ).strftime("%H:%M:%S")
        except Exception:
            display_time = str(raw_time)

        status = str(entry.get("status", "pending")).lower()
        if status in ("early", "late"):
            adjustment = _timing_feedback_adjustment_minutes(
                status,
                entry.get("timing_adjustment_minutes"),
            )
            result_label = (
                f"{adjustment:+d} min"
                if adjustment in (-1, 1)
                else status
            )
        else:
            result_label = status_labels.get(status, status)
        target = entry.get("target", "?")
        max_cote = entry.get("max_cote")
        reached = (
            f"{safe_float(max_cote):.3f}x"
            if max_cote is not None
            else "?"
        )
        rendered.append(
            f"🕒 <code>{html_mod.escape(display_time)}</code> | "
            f"<b>{html_mod.escape(str(entry.get('mode', '?')))}</b> | "
            f"cible {html_mod.escape(str(target))}x | "
            f"atteinte <code>{reached}</code> | "
            f"<b>{result_label}</b>"
        )

    txt = (
        "<b>📜 HISTORIQUE</b>\n\n"
        + "\n".join(rendered[-20:])
    )

    _safe_send_message(
        msg.chat.id,
        txt,
        parse_mode="HTML"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STATS IA
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text == "📊 Stats IA"
)
def show_stats(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    chat_id = msg.chat.id

    total = len(
        feedback_history
    )

    observation_total = len(
        high_cote_observations
    )

    # Les statistiques directionnelles excluent les retours de timing.
    directional_total = sum(
        1
        for f in feedback_history
        if isinstance(f, dict)
        and str(f.get("status") or f.get("learning_status") or "").lower()
            in ("gain", "perte", "loss")
    )

    timing_correction_total = sum(
        1
        for f in feedback_history
        if isinstance(f, dict)
        and str(f.get("status") or f.get("learning_status") or "").lower()
            in ("early", "late")
    )

    measured_feedback = [
        f for f in feedback_history
        if _has_observed_feedback_time(f)
    ]
    measured_timing_total = len(measured_feedback)
    measured_directional_total = sum(
        1
        for f in measured_feedback
        if str(f.get("status") or f.get("learning_status") or "").lower()
            in ("gain", "perte", "loss")
    )
    measured_correction_total = sum(
        1
        for f in measured_feedback
        if str(f.get("status") or f.get("learning_status") or "").lower()
            in ("early", "late")
    )

    wins = sum(
        1
        for f in feedback_history
        if isinstance(f, dict)
        and str(f.get("status") or f.get("learning_status") or "").lower()
            == "gain"
    )

    if (
        total == 0
        and observation_total == 0
    ):

        _safe_send_message(
            chat_id,
            "📊 <b>Aucune statistique disponible</b>",
            parse_mode="HTML"
        )

        return

    rate = (
        wins
        / max(
            directional_total,
            1
        )
    ) * 100

    tg = sum(
        len(
            pattern_memory
            .get("gains", {})
            .get(L, [])
        )
        for L in (
            5,
            10,
            15
        )
    )

    tp = sum(
        len(
            pattern_memory
            .get("pertes", {})
            .get(L, [])
        )
        for L in (
            5,
            10,
            15
        )
    )

    feedback_count = len(
        feedback_samples
    )

    # ══════════════════════════════════════════════════════════════════════
    # CONTEXTE DOMINANT
    # ══════════════════════════════════════════════════════════════════════

    dominant_ctx = "CALM"

    try:

        if last_cotes.get(
            chat_id
        ):

            dominant_ctx = detect_market_context(
                last_cotes[chat_id]
            )

    except Exception:
        pass

    meta_lines = ""

    try:

        meta_w = normalize_meta_weights(
            dominant_ctx
        )

        meta_r = {
            m:
                _get_meta_stats(
                    dominant_ctx,
                    m
                )
            for m in META_MODEL_NAMES
        }

        meta_lines = (
            f"\n<b>🤖 Meta Engine — contexte "
            f"{html_mod.escape(str(dominant_ctx))} :</b>\n"
            +
            "\n".join(
                f"• {m.capitalize()}: "
                f"poids={meta_w.get(m, 0):.1%} "
                f"fiab={meta_r[m].get('reliability', 0):.1%} "
                f"({meta_r[m].get('wins', 0)}W/"
                f"{meta_r[m].get('losses', 0)}L)"
                for m in META_MODEL_NAMES
            )
        )

    except Exception as e:

        logging.warning(
            f"[STATS META] {e}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIENCE RELIABILITY
    # ══════════════════════════════════════════════════════════════════════

    rel_vals = [
        safe_float(
            e.get(
                "reliability",
                EXP_RELIABILITY_DEFAULT
            ),
            EXP_RELIABILITY_DEFAULT
        )

        for e in experience_memory

        if isinstance(
            e,
            dict
        )
    ]

    avg_rel = (
        sum(rel_vals)
        / len(rel_vals)
        if rel_vals
        else EXP_RELIABILITY_DEFAULT
    )

    high_rel = sum(
        1
        for r in rel_vals
        if r >= 0.80
    )

    # ══════════════════════════════════════════════════════════════════════
    # PATH METRIC
    # ══════════════════════════════════════════════════════════════════════

    path_values = [
        safe_float(
            f.get(
                "low_cote_losses"
            ),
            0.0
        )

        for f in feedback_history

        if (
            isinstance(
                f,
                dict
            )
            and f.get(
                "low_cote_losses"
            ) is not None
        )
    ]

    avg_path_losses = (
        sum(path_values)
        / len(path_values)
        if path_values
        else 0.0
    )

    path_observations = len(
        path_values
    )

    # ══════════════════════════════════════════════════════════════════════
    # TIMING
    # ══════════════════════════════════════════════════════════════════════

    timing_lines = []

    for delay in SUPPORTED_DELAYS:

        ds = DELAY_STATS.get(
            delay,
            {}
        )

        timing_lines.append(
            f"• {delay} min : "
            f"{ds.get('success', 0)} corrects / "
            f"{ds.get('loss', 0)} incorrects"
        )

    timing_summary = "\n".join(
        timing_lines
    )

    txt = (
        "<b>📊 STATISTIQUES IA v4.0</b>\n\n"

        "<b>🎯 Performance :</b>\n"
        f"• Feedbacks totaux : {total}\n"
        f"• Retours directionnels : {directional_total} "
        f"({wins} gains / {directional_total - wins} pertes)\n"
        f"• Corrections de timing : {timing_correction_total}\n"
        f"• Horaires réels observés : {measured_timing_total} "
        f"({measured_directional_total} directionnels / "
        f"{measured_correction_total} corrections)\n"
        f"• Taux de réussite : "
        f"<b>{rate:.1f}%</b>\n"
        f"• Observations passives &gt;10x : "
        f"<b>{observation_total}</b>\n\n"

        "<b>⏱️ Timing adaptatif :</b>\n"
        f"{timing_summary}\n\n"

        "<b>🧠 Mémoire :</b>\n"
        f"• Patterns GAIN : {tg} "
        f"| PERTE : {tp}\n"
        f"• SEQ_WEIGHTS : "
        f"5={SEQ_WEIGHTS.get(5, 0):.2f} "
        f"| 10={SEQ_WEIGHTS.get(10, 0):.2f} "
        f"| 15={SEQ_WEIGHTS.get(15, 0):.2f}\n"
        f"• Expériences mémorisées : "
        f"{len(experience_memory)}\n"
        f"• Fiabilité moy. expériences : "
        f"{avg_rel:.1%} "
        f"({high_rel} haute fiabilité)\n"
        f"• Pertes ≤ 1.99x avant issue : "
        f"{avg_path_losses:.1f} en moyenne\n"
        f"  ({path_observations} observations explicites)\n"
        f"• Échantillons de feedback : "
        f"{feedback_count}\n"
        f"• Dataset IA : "
        f"{len(training_dataset)}\n"
        f"• Confiance réseau : "
        f"{neural_confidence() * 100:.1f}%\n"
        f"{meta_lines}"
    )

    _safe_send_message(
        chat_id,
        txt,
        parse_mode="HTML"
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESET DES MÉMOIRES DÉRIVÉES
# ══════════════════════════════════════════════════════════════════════════════

def _reset_derived_learning_memories():
    """
    Nettoie les statistiques dérivées avant reconstruction.

    Cette étape est importante :
    sinon le bouton Réentraînement pouvait compter plusieurs fois les mêmes
    feedbacks.
    """

    global DELAY_STATS

    # ----------------------------------------------------------------------
    # Pattern memory
    # ----------------------------------------------------------------------

    pattern_memory.setdefault(
        "gains",
        {}
    )

    pattern_memory.setdefault(
        "pertes",
        {}
    )

    for key in (
        "gains",
        "pertes"
    ):

        for L in (
            5,
            10,
            15
        ):

            pattern_memory[
                key
            ][L] = []

    # ----------------------------------------------------------------------
    # Timing global
    # ----------------------------------------------------------------------

    DELAY_STATS = {
        delay: {
            "total": 0,
            "success": 0,
            "loss": 0,
            "recent_results": [],
            "avg_score": 0.0,
        }

        for delay in SUPPORTED_DELAYS
    }

    # ----------------------------------------------------------------------
    # Timing contextuel
    # ----------------------------------------------------------------------

    try:

        AI_STATE["timing_memory"] = {}

    except Exception:
        pass

    # ----------------------------------------------------------------------
    # Context memory
    #
    # On ne la vide que si la structure existe et est explicitement une
    # mémoire dérivée reconstruisible.
    # ----------------------------------------------------------------------

    for key in (
        "context_memory",
        "context_stats",
        "model_performance"
    ):

        try:

            obj = globals().get(
                key
            )

            if isinstance(
                obj,
                dict
            ):
                obj.clear()

        except Exception:
            pass

    _normalize_pattern_weights()


def _daily_retrain_worker():
    """
    Lance au plus une fois le réentraînement automatique par date locale.
    Le marqueur est persisté pour éviter un doublon après une reconnexion.
    """
    while True:
        try:
            now = datetime.datetime.now()
            today = now.strftime("%Y-%m-%d")
            last_run = AI_STATE.get("daily_retrain_date")

            if now.hour >= 10 and last_run != today:
                with daily_retrain_lock:
                    # Double vérification après l'acquisition du verrou.
                    if AI_STATE.get("daily_retrain_date") != today:
                        AI_STATE["daily_retrain_date"] = today

                        if len(training_dataset) >= 60:
                            logging.info(
                                "[DAILY RETRAIN] démarrage à %s avec %s échantillons",
                                now.strftime("%H:%M"),
                                len(training_dataset)
                            )
                            _train_model_on_dataset(
                                list(training_dataset)
                            )
                            save_all()
                        else:
                            logging.info(
                                "[DAILY RETRAIN] reporté : %s échantillons, "
                                "minimum 60",
                                len(training_dataset)
                            )
                            save_all()

        except Exception as e:
            logging.error(
                "[DAILY RETRAIN] %s",
                e,
                exc_info=True
            )

        time.sleep(20)


# ══════════════════════════════════════════════════════════════════════════════
# RÉENTRAÎNEMENT MANUEL
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        m.text == "🔄 Réentraînement IA"
)
def manual_retrain(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    chat_id = msg.chat.id

    _safe_send_message(
        chat_id,
        "🔄 <b>Réentraînement complet en cours...</b>",
        parse_mode="HTML"
    )

    try:

        if len(
            training_dataset
        ) < 60:

            _safe_send_message(
                chat_id,
                "⚠️ Minimum 60 feedbacks "
                "directionnels requis pour "
                "réentraîner."
            )

            return

        # ══════════════════════════════════════════════════════════════════
        # 1. RÉENTRAÎNEMENT NEURONAL
        # ══════════════════════════════════════════════════════════════════

        _train_model_on_dataset(
            list(training_dataset)
        )

        # ══════════════════════════════════════════════════════════════════
        # 2. NETTOYAGE DES MÉMOIRES DÉRIVÉES
        # ══════════════════════════════════════════════════════════════════

        _reset_derived_learning_memories()

        # ══════════════════════════════════════════════════════════════════
        # 3. RECONSTRUCTION PATTERNS
        # ══════════════════════════════════════════════════════════════════

        pattern_rebuilt = 0

        for fb in feedback_history:

            if not isinstance(
                fb,
                dict
            ):
                continue

            feedback_learning_status = fb.get(
                "learning_status",
                fb.get("status")
            )

            if feedback_learning_status not in (
                "gain",
                "perte",
                "loss"
            ):
                continue

            pred_data = fb.get(
                "prediction"
            ) or {}

            seq = pred_data.get(
                "pattern_sequence",
                []
            )

            if not seq:
                continue

            update_pattern_memory(
                seq,
                feedback_learning_status
            )

            pattern_rebuilt += 1

        # ══════════════════════════════════════════════════════════════════
        # 4. RECONSTRUCTION CONTEXTES
        # ══════════════════════════════════════════════════════════════════

        ctx_rebuilt = 0

        for fb in feedback_history:

            if not isinstance(
                fb,
                dict
            ):
                continue

            feedback_learning_status = fb.get(
                "learning_status",
                fb.get("status")
            )

            if feedback_learning_status not in (
                "gain",
                "perte",
                "loss"
            ):
                continue

            pred_data = fb.get(
                "prediction"
            ) or {}

            cotes_fb = (
                fb.get(
                    "cotes",
                    []
                )
                or pred_data.get(
                    "cotes",
                    []
                )
                or pred_data.get(
                    "last_cotes",
                    []
                )
            )

            if not cotes_fb:
                continue

            try:

                ctx = detect_market_context(
                    cotes_fb
                )

                pk = build_pattern_key(
                    cotes_fb
                )

                success = feedback_learning_status == "gain"

                confidence = (
                    safe_float(
                        pred_data.get(
                            "pertinence",
                            50.0
                        ),
                        50.0
                    )
                    / 100.0
                )

                update_context_memory(
                    ctx,
                    pk,
                    success,
                    confidence
                )

                ctx_rebuilt += 1

            except Exception as e:

                logging.warning(
                    f"[RETRAIN CONTEXT] {e}"
                )

        # ══════════════════════════════════════════════════════════════════
        # 5. RECONSTRUCTION PATH
        # ══════════════════════════════════════════════════════════════════

        path_rebuilt = 0

        for fb in feedback_history:

            if not isinstance(
                fb,
                dict
            ):
                continue

            if fb.get(
                "low_cote_losses"
            ) is None:

                continue

            exp_id = fb.get(
                "experience_id"
            )

            if not exp_id:
                continue

            for exp in experience_memory:

                if (
                    isinstance(
                        exp,
                        dict
                    )
                    and exp.get(
                        "experience_id"
                    ) == exp_id
                ):

                    exp["low_cote_losses"] = max(
                        0,
                        int(
                            safe_float(
                                fb.get(
                                    "low_cote_losses",
                                    0
                                ),
                                0
                            )
                        )
                    )

                    path_rebuilt += 1

                    break

        # ══════════════════════════════════════════════════════════════════
        # 6. RECONSTRUCTION TIMING — SOURCE TEMPORELLE UNIQUEMENT
        # ══════════════════════════════════════════════════════════════════

        delay_rebuilt = rebuild_delay_stats_from_feedback()

        # Rebuild explicit hot/weak/fall/rise/transition families from the
        # retained feedback ledger; this replaces only this new derived namespace.
        rebuild_pattern_timing_memory()
        rebuild_sequence_feature_memory()

        # ══════════════════════════════════════════════════════════════════
        # 7. NORMALISATION
        # ══════════════════════════════════════════════════════════════════

        _normalize_pattern_weights()

        # ══════════════════════════════════════════════════════════════════
        # 8. SAUVEGARDE
        # ══════════════════════════════════════════════════════════════════

        save_all()

        # ══════════════════════════════════════════════════════════════════
        # 9. STATISTIQUES
        # ══════════════════════════════════════════════════════════════════

        wins = sum(
            1
            for f in feedback_history
            if f.get(
                "status"
            ) == "gain"
        )

        # Le taux directionnel exclut strictement les feedbacks de timing.
        directional_total = sum(
            1
            for f in feedback_history
            if isinstance(f, dict)
            and str(f.get("status", f.get("learning_status", ""))).lower()
                in ("gain", "perte", "loss")
        )

        rate = (
            wins
            / max(
                directional_total,
                1
            )
        ) * 100

        txt = (
            "✅ <b>RÉENTRAÎNEMENT COMPLET !</b>\n\n"

            f"• Dataset : "
            f"{len(training_dataset)} échantillons\n"

            f"• Réseau neuronal : "
            f"{neural_confidence() * 100:.1f}% confiance\n"

            f"• Patterns reconstruits : "
            f"{pattern_rebuilt}\n"

            f"• Contextes reconstruits : "
            f"{ctx_rebuilt}\n"

            f"• Parcours ≤ 1.99x reconstruits : "
            f"{path_rebuilt}\n"

            f"• Délais reconstruits : "
            f"{delay_rebuilt}\n"

            f"• SEQ_WEIGHTS : "
            f"5={SEQ_WEIGHTS.get(5, 0):.3f} "
            f"| 10={SEQ_WEIGHTS.get(10, 0):.3f} "
            f"| 15={SEQ_WEIGHTS.get(15, 0):.3f}\n"

            f"• Taux de gain historique : "
            f"{rate:.1f}% "
            f"({wins}/{directional_total})\n"

            f"• Expériences en mémoire : "
            f"{len(experience_memory)}\n\n"

            f"<i>Les mémoires dérivées ont été "
            f"reconstruites sans double comptage.</i>"
        )

        _safe_send_message(
            chat_id,
            txt,
            parse_mode="HTML"
        )

    except Exception as e:

        logging.error(
            f"[RETRAIN] {e}",
            exc_info=True
        )

        _safe_send_message(
            chat_id,
            f"❌ <b>Erreur réentraînement :</b>\n"
            f"<code>{html_mod.escape(str(e))}</code>",
            parse_mode="HTML"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHOTO / OCR
# ══════════════════════════════════════════════════════════════════════════════

def _format_extracted_cotes(cotes, normal_cotes, isolated_high):
    """Affiche chaque cote détectée dans son ordre d'apparition."""
    visible_limit = 40
    lines = []

    for index, cote in enumerate(cotes[:visible_limit], start=1):
        value = float(cote)
        displayed = f"{value:.2f}"
        marker = (
            " — <i>exclue du calcul (&gt;10x)</i>"
            if value > 10.0
            else ""
        )
        lines.append(f"{index}. <code>{displayed}x</code>{marker}")

    if len(cotes) > visible_limit:
        lines.append(
            f"… et {len(cotes) - visible_limit} autre(s) cote(s)."
        )

    message = (
        f"📸 <b>Cotes extraites dans l'ordre ({len(cotes)})</b>\n"
        + "\n".join(lines)
        + f"\n\n✅ <b>{len(normal_cotes)} cote(s) ≤10x "
        "seront utilisées pour le calcul.</b>"
    )

    if isolated_high:
        message += (
            f"\n🧱 {len(isolated_high)} cote(s) &gt;10x "
            "affichée(s), mais exclue(s) du calcul."
        )

    return message


@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(msg):

    chat_id = msg.chat.id
    photos = getattr(msg, "photo", None) or []
    if not photos:
        return

    photo_identity = (
        getattr(msg, "message_id", None)
        or getattr(photos[-1], "file_id", None)
    )

    if is_duplicate_message(
        chat_id,
        f"photo:{photo_identity}"
    ):
        return

    path = None

    try:

        fi = bot.get_file(
            msg.photo[-1].file_id
        )

        data = bot.download_file(
            fi.file_path
        )

        os.makedirs(
            TEMP_DIR,
            exist_ok=True
        )

        filename = (
            f"{chat_id}_"
            f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            f".png"
        )

        path = os.path.join(
            TEMP_DIR,
            filename
        )

        with open(
            path,
            "wb"
        ) as f:

            f.write(
                data
            )

        cotes = extract_cotes(
            path,
            debug=True
        )

        if cotes:

            normal_cotes, isolated_high = split_analysis_cotes(
                cotes
            )

            extraction_message = _format_extracted_cotes(
                cotes,
                normal_cotes,
                isolated_high
            )

            if len(normal_cotes) < 5:

                last_cotes.pop(chat_id, None)

                _safe_send_message(
                    chat_id,
                    extraction_message
                    + "\n\n⚠️ Il faut au moins 5 cotes "
                    "normales pour lancer une prédiction.",
                    parse_mode="HTML",
                )

                return

            last_cotes[
                chat_id
            ] = list(
                cotes
            )

            _safe_send_message(
                chat_id,
                extraction_message
                + "\n\nChoisissez votre mode :",
                parse_mode="HTML",
                reply_markup=main_keyboard()
            )

        else:

            last_cotes.pop(chat_id, None)

            _safe_send_message(
                chat_id,

                "⚠️ Aucune cote détectée.\n\n"
                "Essayez une capture plus nette "
                "ou envoyez les cotes en texte."
            )

    except Exception as e:

        _safe_send_message(
            chat_id,
            f"❌ <b>Erreur image :</b>\n"
            f"<code>{html_mod.escape(str(e))}</code>",
            parse_mode="HTML"
        )

        logging.error(
            f"[PHOTO] {e}",
            exc_info=True
        )

    finally:

        # IMPORTANT :
        # suppression du fichier après OCR.
        if path:

            try:

                if os.path.exists(
                    path
                ):
                    os.remove(
                        path
                    )

            except Exception as e:

                logging.warning(
                    f"[PHOTO CLEANUP] {e}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# TEXTES RÉSERVÉS
# ══════════════════════════════════════════════════════════════════════════════

RESERVED_TEXTS = {
    "🎯 2X-10X",
    "🚀 Grosse Cote",
    "🔮 Ultra Grosse Cote",
    "📜 Historique",
    "📊 Stats IA",
    "🔄 Réentraînement IA",
    "📝 Observer >10x",
    "📋 Heures >10x",
}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRÉE TEXTE DES COTES
# ══════════════════════════════════════════════════════════════════════════════

@bot.message_handler(
    func=lambda m:
        bool(
            m.text
            and re.search(
                r"\d",
                m.text
            )
            and m.text not in RESERVED_TEXTS
        )
)
def handle_text_cotes(msg):

    if is_duplicate_message(
        msg.chat.id,
        msg.text
    ):
        return

    chat_id = msg.chat.id

    try:

        text = str(
            msg.text or ""
        )

        # Uniformisation
        text = (
            text
            .replace(",", ".")
            .replace("x", " ")
            .replace("X", " ")
            .replace("×", " ")
        )

        cotes = []

        for tok in re.findall(
            r"\d+(?:\.\d+)?",
            text
        ):

            try:

                value = float(
                    tok
                )

                if (
                    1.0
                    <= value
                    <= 1000.0
                ):

                    cotes.append(
                        round(
                            value,
                            3
                        )
                    )

            except Exception:
                continue

        normal_cotes, isolated_high = split_analysis_cotes(
            cotes
        )

        if len(normal_cotes) < 5:

            _safe_send_message(
                chat_id,

                "❌ Minimum 5 cotes normales.\n\n"
                "Exemple : "
                "<code>1.5 2.0 3.5 1.8 4.2</code>",

                parse_mode="HTML"
            )

            return

        last_cotes[
            chat_id
        ] = list(
            cotes
        )

        isolated_info = ""

        if isolated_high:

            isolated_info = (
                f"\n🧱 {len(isolated_high)} "
                f"cote(s) &gt;10x isolée(s) "
                f"des statistiques normales."
            )

        _safe_send_message(
            chat_id,

            f"✅ <b>{len(normal_cotes)} "
            f"cotes enregistrées !</b>"
            f"{isolated_info}\n\n"
            f"Choisissez votre mode :",

            parse_mode="HTML",
            reply_markup=main_keyboard()
        )

    except Exception as e:

        logging.error(
            f"[TEXT_COTES] {e}",
            exc_info=True
        )

        _safe_send_message(
            chat_id,
            "❌ Impossible de traiter les cotes."
        )


# ══════════════════════════════════════════════════════════════════════════════
# FLASK — WEBHOOK / HEALTH
# ══════════════════════════════════════════════════════════════════════════════

WEBHOOK_URL_BASE = os.getenv(
    "WEBHOOK_URL_BASE",
    ""
).strip()

WEBHOOK_PATH = (
    "/telegram/"
    + hashlib.sha256(BOT_TOKEN.encode("utf-8")).hexdigest()
)


@flask_app.route(
    WEBHOOK_PATH,
    methods=["POST"]
)
def webhook():

    try:

        raw_data = request.get_data()

        if not raw_data:
            return "", 400

        update = telebot.types.Update.de_json(
            raw_data.decode(
                "utf-8"
            )
        )

        if update is None:
            return "", 400

        bot.process_new_updates(
            [update]
        )

        return "", 200

    except Exception as e:

        logging.warning(
            f"[WEBHOOK] {e}",
            exc_info=True
        )

        return "", 500


@flask_app.route(
    "/"
)
def index():

    return (
        f"✅ Rocket Queen V7 — IA Lucky Jet "
        f"— expériences={len(experience_memory)} "
        f"feedbacks={len(feedback_history)} "
        f"dataset={len(training_dataset)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ V6 — DATA-CALIBRATED TIMING LAYER                                 ║
# ║ Intégrée au moteur V6 principal.                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

V6_TIMING_MIN_HISTORY = 30
V6_TIMING_MAX_DELAY_SEC = max(SUPPORTED_DELAYS) * 60 + 59
V6_TIMING_MAX_OBSERVED_SEC = 30 * 60
V6_TIMING_INTERVAL_COVERAGE = 0.80
V6_TIMING_MIN_BUCKET_SUPPORT = 8
V6_TIMING_ABS_ERROR_CAP_SEC = 3 * 60


def _v6_delay_value(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x <= 0 or x > V6_TIMING_MAX_DELAY_SEC:
        return None
    return x


def _v6_prediction_hour(exp):
    for key in ("prediction_datetime", "predicted_datetime", "predicted_time", "created_at", "timestamp", "pred_time"):
        raw = exp.get(key)
        if not raw:
            continue
        try:
            text = str(raw)
            if "T" in text:
                return int(text[11:13])
            if len(text) >= 2 and text[:2].isdigit():
                return int(text[:2])
        except Exception:
            pass
    return None


def _v6_bucket(seconds):
    value = _v6_delay_value(seconds)
    if value is None or value < 60.0:
        return None
    bucket = int(value // 60.0)
    return bucket if bucket in SUPPORTED_DELAYS else None


def _v6_verified_timing_rows():
    """Read verified timing observations with explicit minute/second handling."""
    rows = []
    for exp in (experience_memory or []):
        if not isinstance(exp, dict):
            continue
        status = str(exp.get("outcome", exp.get("status", ""))).lower().strip()
        if status == "pending" or status in ("early", "late"):
            # Legacy early/late rows may contain inferred/clamped delays, not
            # measured event times; they must not calibrate actual-delay models.
            continue
        has_observed_time = bool(exp.get("actual_time") or exp.get("actual_datetime") or exp.get("real_time"))
        if status in ("gain", "perte", "loss") and not has_observed_time:
            continue

        pred = None
        for key in ("predicted_delay_sec", "pred_delay_sec"):
            try:
                value = float(exp.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 0 < value <= V6_TIMING_MAX_DELAY_SEC:
                pred = value
                break
        if pred is None:
            for key in ("predicted_delay", "delay_min", "predicted_delay_minutes"):
                try:
                    value = float(exp.get(key))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value) or value <= 0:
                    continue
                if key.endswith("_min") or value <= 3.0:
                    value *= 60.0
                if 0 < value <= V6_TIMING_MAX_DELAY_SEC:
                    pred = value
                    break

        actual = None
        for key in ("actual_delay_sec", "timing_learning_delay_sec", "delay_seconds"):
            try:
                value = float(exp.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 0 < value <= V6_TIMING_MAX_OBSERVED_SEC:
                actual = value
                break
        if actual is None:
            for key in ("actual_delay_min", "actual_delay"):
                try:
                    value = float(exp.get(key))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value) or value <= 0:
                    continue
                if key.endswith("_min") or value <= 3.0:
                    value *= 60.0
                if 0 < value <= V6_TIMING_MAX_OBSERVED_SEC:
                    actual = value
                    break

        if pred is None or actual is None:
            continue
        rows.append({
            "pred": pred,
            "actual": actual,
            "error": actual - pred,
            "mode": exp.get("mode"),
            "hour": _v6_prediction_hour(exp),
        })
    return rows


def _v6_normalize_distribution(d):
    keys=list(SUPPORTED_DELAYS)
    vals={int(k):max(0.0,_safe_float(d.get(k,d.get(str(k),0.0)),0.0)) for k in keys}
    total=sum(vals.values())
    if total<=0:
        return {k:1.0/len(keys) for k in keys}
    return {k:vals[k]/total for k in keys}


V6_TIMING_CALIBRATION={
    "status":"untrained",
    "n":0,
    "global_actual_bucket":{k:1.0/len(SUPPORTED_DELAYS) for k in SUPPORTED_DELAYS},
    "conditional_bucket":{},
    "residual_abs_quantile":60.0,
    "coverage":{},
}


def _v6_build_timing_calibration():
    rows=_v6_verified_timing_rows()
    calibration={
        "status":"untrained","n":len(rows),
        "global_actual_bucket":{k:1.0/len(SUPPORTED_DELAYS) for k in SUPPORTED_DELAYS},
        "conditional_bucket":{},"residual_abs_quantile":60.0,"coverage":{},
    }
    if len(rows)<V6_TIMING_MIN_HISTORY:
        return calibration
    global_counts={k:1.0 for k in SUPPORTED_DELAYS}
    cond={k:{j:1.0 for j in SUPPORTED_DELAYS} for k in SUPPORTED_DELAYS}
    errors=[]
    for r in rows:
        pb = _v6_bucket(r["pred"])
        if pb is None:
            continue
        # Long delays outside the allowed 1–3 minute choices still count as
        # timing errors, but are not falsely relabeled as a successful 3-min bucket.
        errors.append(abs(r["error"]))
        ab = _v6_bucket(r["actual"])
        if ab is not None:
            global_counts[ab] += 1.0
            cond[pb][ab] += 1.0
    calibration["global_actual_bucket"]=_v6_normalize_distribution(global_counts)
    calibration["conditional_bucket"]={pb:_v6_normalize_distribution(d) for pb,d in cond.items()}
    if errors:
        errors.sort(); idx=min(len(errors)-1,max(0,int(math.ceil(.80*len(errors)))-1))
        calibration["residual_abs_quantile"]=round(errors[idx],3)
        calibration["coverage"]={str(w):round(sum(e<=w for e in errors)/len(errors),4) for w in (30,60,90,120)}
    calibration["status"]="trained"
    return calibration


def refresh_v6_timing_calibration():
    global V6_TIMING_CALIBRATION
    V6_TIMING_CALIBRATION=_v6_build_timing_calibration()
    logging.info("[V6 TIMING] calibration status=%s n=%s coverage=%s",V6_TIMING_CALIBRATION.get("status"),V6_TIMING_CALIBRATION.get("n"),V6_TIMING_CALIBRATION.get("coverage"))
    return V6_TIMING_CALIBRATION


def v6_timing_calibration_report():
    return dict(V6_TIMING_CALIBRATION)


def _v6_calibrate_delay_distribution(base_distribution,current_delay_sec=None):
    base=_v6_normalize_distribution(base_distribution or {})
    cal=V6_TIMING_CALIBRATION
    if cal.get("status")!="trained":
        return base
    predicted_bucket = _v6_bucket(current_delay_sec) if current_delay_sec is not None else None
    if predicted_bucket in SUPPORTED_DELAYS:
        conditional = cal.get("conditional_bucket", {}).get(predicted_bucket) or cal.get("global_actual_bucket", {})
    else:
        # No selected delay exists yet: use only the global calibration prior.
        conditional = cal.get("global_actual_bucket", {})
    blended={k:.65*base.get(k,0.0)+.35*conditional.get(k,0.0) for k in SUPPORTED_DELAYS}
    return _v6_normalize_distribution(blended)


def _v6_timing_interval_from_residual(center_sec,coverage=V6_TIMING_INTERVAL_COVERAGE):
    q=clamp(_safe_float(V6_TIMING_CALIBRATION.get("residual_abs_quantile"),60.0),15.0,60.0)
    return {"low_sec":round(max(0.0,center_sec-q),1),"high_sec":round(min(V6_TIMING_MAX_DELAY_SEC,center_sec+q),1),"coverage_target":round(coverage,3),"calibration_n":int(V6_TIMING_CALIBRATION.get("n",0)),"method":"v6_historical_residual_quantile"}


def v6_backtest_timing(experiences=None):
    """Backtest chronologique du timing contre l'ancien choix et des baselines.

    Quand feedback_history contient les delay_scores sauvegardés au moment de
    chaque prédiction, rejoue la politique adaptative sur ces seules preuves
    pré-prédiction. Les observations tardives jusqu’à 30 minutes sont comptées comme erreurs; les horodatages aberrants sont exclus.
    """
    source = experiences if experiences is not None else feedback_history
    rows = []

    def _read_seconds(record, second_keys, minute_keys, allow_long=False):
        for key in second_keys:
            try:
                value = float(record.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0 and (allow_long or value <= 600):
                return value
        for key in minute_keys:
            try:
                value = float(record.get(key))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0:
                continue
            if key.endswith("_min") or value <= 3.0:
                value *= 60.0
            if value > 0 and (allow_long or value <= 600):
                return value
        return None

    for index, exp in enumerate(source or []):
        if not isinstance(exp, dict):
            continue
        if source is not None or exp in (feedback_history or []):
            status = str(exp.get("learning_status", exp.get("status", exp.get("outcome", ""))) or "").lower().strip()
            if status in ("early", "late"):
                continue
            if status in ("gain", "perte", "loss"):
                measured = exp.get("actual_delay_source") == "observed_datetime"
                # Backward compatibility for older records with explicit real timestamp.
                if not measured:
                    measured = bool(exp.get("actual_datetime") or exp.get("actual_time") or exp.get("real_time"))
                if not measured:
                    continue
        predicted = _read_seconds(
            exp,
            ("predicted_delay_sec", "pred_delay_sec"),
            ("predicted_delay", "delay_min", "predicted_delay_minutes"),
        )
        actual = _read_seconds(
            exp,
            ("actual_delay_sec", "timing_learning_delay_sec", "delay_seconds"),
            ("actual_delay_min", "actual_delay"),
            allow_long=True,
        )
        # Une mesure de feedback temporel doit être dans la fenêtre réelle
        # du bot (1-3 min). Les valeurs proches de 24 h sont des dates mal
        # interprétées et ne doivent jamais entrer dans le backtest.
        if (predicted is None or actual is None or predicted < 45
                or predicted > V6_TIMING_MAX_DELAY_SEC
                or actual <= 0 or actual > V6_TIMING_MAX_OBSERVED_SEC):
            continue
        raw_scores = exp.get("delay_scores", {})
        scores = {}
        if isinstance(raw_scores, dict):
            for delay in SUPPORTED_DELAYS:
                item = raw_scores.get(delay, raw_scores.get(str(delay), {}))
                if isinstance(item, dict) and item:
                    scores[delay] = dict(item)
        stamp = exp.get("pred_time") or exp.get("predicted_datetime") or exp.get("created_at") or exp.get("time")
        try:
            stamp_key = datetime.datetime.fromisoformat(str(stamp)).timestamp()
        except Exception:
            stamp_key = float(index)
        rows.append({"pred": predicted, "actual": actual, "scores": scores, "stamp": stamp_key})

    rows.sort(key=lambda row: row["stamp"])
    if len(rows) < 5:
        return {"status": "insufficient_data", "n": len(rows), "minimum_for_descriptive_report": 5, "minimum_for_calibration": V6_TIMING_MIN_HISTORY}

    cut = max(1, int(len(rows) * 0.80))
    test = rows[cut:]
    if not test:
        return {"status": "insufficient_data", "n": len(rows), "minimum": V6_TIMING_MIN_HISTORY}

    historical_errors = [abs(row["actual"] - row["pred"]) for row in test]
    historical_mae = sum(historical_errors) / len(historical_errors)
    baseline_mae = {
        str(delay): round(sum(abs(row["actual"] - _delay_center(delay)) for row in test) / len(test), 3)
        for delay in SUPPORTED_DELAYS
    }
    best_fixed = min(baseline_mae, key=baseline_mae.get)

    # Replay the adaptive selection only where the prediction-time score vector
    # was actually saved. A fixed seed makes the diagnostic reproducible.
    replay_rows = [row for row in test if all(delay in row["scores"] for delay in SUPPORTED_DELAYS)]
    # A single random draw is a noisy and misleading backtest for an exploration
    # policy. Estimate its expected absolute error from the policy probabilities;
    # also report deterministic argmax error as a separate diagnostic.
    adaptive_mae = None
    adaptive_argmax_mae = None
    adaptive_n = len(replay_rows)
    if replay_rows:
        expected_errors = []
        argmax_errors = []
        for row in replay_rows:
            probs, spread, total_support, min_support = _timing_selection_probabilities(row["scores"])
            if min_support < 8 or spread < 0.035:
                expected_error = sum(
                    probs.get(delay, 0.0) * abs(row["actual"] - _delay_center(delay))
                    for delay in SUPPORTED_DELAYS
                )
            else:
                selected = max(SUPPORTED_DELAYS, key=lambda delay: probs.get(delay, 0.0))
                expected_error = abs(row["actual"] - _delay_center(selected))
            argmax_delay = max(SUPPORTED_DELAYS, key=lambda delay: probs.get(delay, 0.0))
            expected_errors.append(expected_error)
            argmax_errors.append(abs(row["actual"] - _delay_center(argmax_delay)))
        adaptive_mae = round(sum(expected_errors) / len(expected_errors), 3)
        adaptive_argmax_mae = round(sum(argmax_errors) / len(argmax_errors), 3)

    abs_errors = historical_errors
    median_abs = sorted(abs_errors)[len(abs_errors) // 2]
    coverage = {str(window): round(sum(error <= window for error in abs_errors) / len(abs_errors), 4) for window in (30, 60, 90, 120)}
    return {
        "status": ("ok" if len(rows) >= V6_TIMING_MIN_HISTORY else "descriptive_only_insufficient_data"),
        "minimum_for_calibration": V6_TIMING_MIN_HISTORY,
        "method": "chronological_holdout_with_saved_score_replay",
        "n": len(rows),
        "train_n": cut,
        "validation_n": len(test),
        "historical_selected_mae_seconds": round(historical_mae, 3),
        "median_absolute_error_seconds": round(median_abs, 3),
        "within_30_seconds_rate": round(sum(error <= 30.0 for error in abs_errors) / len(abs_errors), 4),
        "coverage": coverage,
        "fixed_delay_baseline_mae_seconds": baseline_mae,
        "best_fixed_delay_baseline": int(best_fixed),
        "historical_beats_best_fixed_baseline": bool(historical_mae < baseline_mae[best_fixed]),
        "adaptive_replay_n": adaptive_n,
        "adaptive_replay_mae_seconds": adaptive_mae,
        "adaptive_expected_policy_mae_seconds": adaptive_mae,
        "adaptive_argmax_mae_seconds": adaptive_argmax_mae,
        "adaptive_replay_beats_historical": (bool(adaptive_mae < historical_mae) if adaptive_mae is not None else None),
        "adaptive_expected_policy_beats_best_fixed_baseline": (bool(adaptive_mae < baseline_mae[best_fixed]) if adaptive_mae is not None else None),
        "note": "Le replay d'exploration utilise l'erreur absolue attendue selon les probabilités de la politique, et non un tirage aléatoire unique. Les scores sauvegardés doivent être ceux disponibles avant la prédiction; un petit holdout reste descriptif.",
    }


# This registry is the single runtime entry point. Older persisted memories
# can still provide evidence, but they cannot replace the functions from the
# script currently installed in main.py.
PRIMARY_ENGINE = {
    "script_id": PRIMARY_SCRIPT_ID,
    "script_version": PRIMARY_SCRIPT_VERSION,
    "run_prediction": run_prediction,
    "build_current_experience": build_current_experience,
    "compute_memory_learning": compute_memory_learning,
    "memory_based_prediction": memory_based_prediction,
    "v6_timing_calibration": v6_timing_calibration_report,
}


if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════════════════
    # CHARGEMENT MÉMOIRE
    # ══════════════════════════════════════════════════════════════════════

    load_all()

    # Les historiques sont désormais disponibles : calibrer la V6 sur
    # les expériences réellement chargées, et non sur la mémoire vide
    # présente au moment de l'import du module.
    refresh_v6_timing_calibration()

    # Exécuter les diagnostics historiques au démarrage. Ils n'entraînent pas
    # le modèle de production et ne modifient pas les mémoires. Les rapports
    # indiquent explicitement quand l'échantillon est insuffisant.
    try:
        direction_report = v6_backtest_direction(training_dataset)
        logging.info("[HISTORICAL BACKTEST][DIRECTION] %s", direction_report)
    except Exception as exc:
        logging.warning("[HISTORICAL BACKTEST][DIRECTION] %s", exc)
    try:
        directional_feedback_report = v6_backtest_directional_feedback(feedback_history)
        logging.info("[HISTORICAL BACKTEST][DIRECTIONAL FEEDBACK] %s", directional_feedback_report)
    except Exception as exc:
        logging.warning("[HISTORICAL BACKTEST][DIRECTIONAL FEEDBACK] %s", exc)
    try:
        timing_report = v6_backtest_timing(feedback_history)
        logging.info("[HISTORICAL BACKTEST][TIMING] %s", timing_report)
    except Exception as exc:
        logging.warning("[HISTORICAL BACKTEST][TIMING] %s", exc)
    try:
        level_report = v6_backtest_prediction_levels(feedback_samples)
        logging.info("[HISTORICAL BACKTEST][LEVELS] %s", level_report)
    except Exception as exc:
        logging.warning("[HISTORICAL BACKTEST][LEVELS] %s", exc)
    try:
        quality_report = v6_historical_data_quality_report(feedback_samples, feedback_history)
        logging.info("[HISTORICAL DATA QUALITY] %s", quality_report)
    except Exception as exc:
        logging.warning("[HISTORICAL DATA QUALITY] %s", exc)

    # Sécurité après chargement
    _normalize_pattern_weights()

    logging.info(
        "[PRIMARY] bot_instances=1 script=%s version=%s prediction=%s data_dir=%s",
        PRIMARY_ENGINE["script_id"],
        PRIMARY_ENGINE["script_version"],
        PRIMARY_ENGINE["run_prediction"].__name__,
        DATA_DIR,
    )

    # Automatic retraining is driven by every 60th definitive feedback.
    # We do not start a second daily retraining loop: it could overlap with
    # feedback-triggered training and make model updates race each other.

    logging.info(
        "🚀 Rocket Queen V7 démarré "
        "(Modules mémoire + timing + feedback actifs)"
    )

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    # ══════════════════════════════════════════════════════════════════════
    # POLLING
    # ══════════════════════════════════════════════════════════════════════

    if not WEBHOOK_URL_BASE:

        try:

            bot.remove_webhook()

        except Exception as e:

            logging.warning(
                f"[START] remove_webhook: {e}"
            )

        # Flask reste actif pour le health-check Replit.
        threading.Thread(
            target=lambda:
                flask_app.run(
                    host="0.0.0.0",
                    port=port,
                    use_reloader=False
                ),
            daemon=True
        ).start()

        while True:

            try:

                logging.info(
                    "[POLLING] Démarrage du long-polling..."
                )

                bot.infinity_polling(
                    timeout=60,
                    long_polling_timeout=60,
                    logger_level=logging.ERROR,
                    allowed_updates=[
                        "message",
                        "callback_query"
                    ],
                )

            except KeyboardInterrupt:

                logging.info(
                    "[POLLING] Arrêt manuel."
                )

                try:
                    save_all()
                except Exception:
                    pass

                break

            except Exception as e:

                err_str = str(
                    e
                )

                try:
                    save_all()
                except Exception:
                    pass

                # ----------------------------------------------------------
                # Une autre instance Telegram utilise déjà le bot.
                # ----------------------------------------------------------

                if (
                    "409" in err_str
                    or "Conflict" in err_str
                    or "terminated by other getUpdates request"
                    in err_str
                ):

                    logging.error(
                        "[POLLING] Conflit 409 — "
                        "une autre instance est active. "
                        "Arrêt."
                    )

                    break

                # ----------------------------------------------------------
                # Erreur réseau : reconnexion
                # ----------------------------------------------------------

                logging.warning(
                    "[POLLING] Erreur réseau, "
                    "reconnexion dans 5 secondes : %s",
                    e
                )

                time.sleep(
                    5
                )

    # ══════════════════════════════════════════════════════════════════════
    # WEBHOOK
    # ══════════════════════════════════════════════════════════════════════

    else:

        try:

            bot.remove_webhook()

            time.sleep(
                0.5
            )

            bot.set_webhook(
                url=(
                    f"{WEBHOOK_URL_BASE}"
                    f"{WEBHOOK_PATH}"
                )
            )

            logging.info(
                "[WEBHOOK] Activé : %s%s",
                WEBHOOK_URL_BASE,
                WEBHOOK_PATH
            )

            flask_app.run(
                host="0.0.0.0",
                port=port
            )

        except Exception as e:

            logging.error(
                f"[WEBHOOK START] {e}",
                exc_info=True
            )

            try:
                save_all()
            except Exception:
                pass
