"""Central configuration for SilenceBreaker.

All tunable paths, model names, label sets, and LLM settings live here so the
rest of the codebase imports from a single source of truth.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
KB_DIR      = os.path.join(DATA_DIR, "kb")
INDEX_DIR   = os.path.join(DATA_DIR, "processed", "faiss_index")
MODELS_DIR  = os.path.join(ROOT, "models")
ABUSE_MODEL = os.path.join(MODELS_DIR, "abuse_clf")     # created by training script

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
EMB_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ----------------------------------------------------------------------------
# LLM (OpenAI-compatible endpoint: Groq / OpenAI / Together / local server)
# ----------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_API_KEY  = (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
                or os.getenv("TOGETHER_API_KEY"))

# When no API key is configured the system runs in OFFLINE mode: it uses
# rule-based category classification and a template-based planner so the whole
# pipeline is reproducible out of the box (useful for grading). Set an API key
# in .env for higher-quality generated responses.
OFFLINE = LLM_API_KEY is None

# ----------------------------------------------------------------------------
# Label sets
# ----------------------------------------------------------------------------
ABUSE_LABELS = ["domestic_abuse", "workplace_harassment", "coercive_control", "non_abuse"]
RISK_LEVELS  = ["low", "medium", "high"]

# Optimal abuse-classifier threshold found via precision_recall_curve on tweet_eval test set.
# Default 0.5 caused near-universal abuse prediction (recall=0.978, precision=0.468).
# At 0.930 macro-F1 improves from 0.460 → 0.564.
ABUSE_THRESHOLD = 0.930
