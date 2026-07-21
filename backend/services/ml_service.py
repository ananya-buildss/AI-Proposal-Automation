import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

embedder = joblib.load(os.path.join(MODEL_DIR, "embedder.pkl"))

domain_model = joblib.load(os.path.join(MODEL_DIR, "domain_classifier.pkl"))
complexity_model = joblib.load(os.path.join(MODEL_DIR, "complexity_classifier.pkl"))
timeline_model = joblib.load(os.path.join(MODEL_DIR, "timeline_classifier.pkl"))
budget_model = joblib.load(os.path.join(MODEL_DIR, "budget_classifier.pkl"))


def analyze_requirement(text):

    embedding = embedder.encode([text])

    return {
        "domain": domain_model.predict(embedding)[0],
        "complexity": complexity_model.predict(embedding)[0],
        "timeline": timeline_model.predict(embedding)[0],
        "budget": budget_model.predict(embedding)[0]
    }