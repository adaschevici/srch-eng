from typing import List
from torch import Tensor
from numpy import ndarray

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentences: List[str]) -> List[Tensor] | Tensor | ndarray:
        return self.model.encode(sentences)

    def embed_query(self, text: str) -> List[float]:
        return list(map(float, self.model.encode(text)))
