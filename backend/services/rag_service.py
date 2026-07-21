import os
import chromadb
from sentence_transformers import SentenceTransformer

# ===========================
# Configuration
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "ml", "vector_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "projects"

# ===========================
# Load Model
# ===========================

model = SentenceTransformer(EMBEDDING_MODEL)

# ===========================
# Connect Database
# ===========================

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    raise RuntimeError(
        f"Unable to load collection '{COLLECTION_NAME}'.\n{e}"
    )


# ===========================
# Retrieve Similar Projects
# ===========================

def retrieve_similar_projects(requirement: str, top_k: int = 3):

    if not requirement.strip():
        return []

    embedding = model.encode(requirement).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas"]
    )

    projects = []

    for meta in results["metadatas"][0]:

        projects.append({

            "title": meta["title"],

            "domain": meta["domain"],

            "summary": meta["summary"],

            "tech_stack": meta["tech_stack"]

        })

    return projects