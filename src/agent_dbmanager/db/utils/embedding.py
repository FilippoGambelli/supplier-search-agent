from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    return model.encode(text).tolist()