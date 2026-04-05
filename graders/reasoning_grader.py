import sys
import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import warnings

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Load model once at module level to avoid reloading
    # all-MiniLM-L6-v2 is ~80MB, very fast for CPU
    model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    import difflib


def grade_reasoning(predicted_reasoning: str, gold_justification: str) -> float:
    """
    Grade reasoning quality using semantic similarity (sentence-transformers).
    Falls back to difflib if sentence-transformers is not available.

    Args:
        predicted_reasoning: Agent's reasoning string
        gold_justification: True justification

    Returns:
        Reward (0.0-1.0)
    """
    pred = predicted_reasoning.strip()
    gold = gold_justification.strip()

    if not pred or not gold:
        return 0.1

    if HAS_MODEL:
        try:
            # Compute embeddings
            embeddings = model.encode([pred, gold])

            # Compute cosine similarity
            # embeddings are normalized by default in many ST models, but let's be safe
            norm_0 = np.linalg.norm(embeddings[0])
            norm_1 = np.linalg.norm(embeddings[1])

            if norm_0 == 0 or norm_1 == 0:
                return 0.1

            sim = np.dot(embeddings[0], embeddings[1]) / (norm_0 * norm_1)

            # Semantic similarity can be negative, clip to 0-1
            # Usually, good matches are > 0.6
            ratio = float(max(0.0, sim))
        except Exception as e:
            print(f"[WARN] Error in reasoning grader: {e}", file=sys.stderr)
            ratio = 0.0
    else:
        # Fallback to simple difflib
        pred_lower = pred.lower()
        gold_lower = gold.lower()
        seq = difflib.SequenceMatcher(None, pred_lower, gold_lower)
        ratio = seq.ratio()

    # Smooth the curve
    if ratio > 0.7:
        return 1.0
    elif ratio > 0.5:
        return 0.8
    elif ratio > 0.3:
        return 0.5
    elif ratio > 0.1:
        return 0.2
    else:
        return 0.0


if __name__ == "__main__":
    r1 = grade_reasoning("Contains toxic insults targeting a group", "Toxic insults")
    print(f"Good match: {r1}")
    r2 = grade_reasoning("Safe and wholesome content", "Toxic insults")
    print(f"Bad match: {r2}")
