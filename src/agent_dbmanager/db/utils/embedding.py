from config import MODEL as model

def get_embedding(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    return model.encode(text).tolist()