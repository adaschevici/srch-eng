import time

from langchain_community.vectorstores import FAISS
from ray import serve
from starlette.requests import Request

from embeddings import LocalHuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index_fast"


@serve.deployment
class VectorSearchDeployment:
    def __init__(self):
        st = time.time()
        self.embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
        self.db = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
        et = time.time() - st
        print(f"Database loaded in {et:.2f} seconds")

    def search(self, query: str):
        results = self.db.max_marginal_relevance_search(query)
        retval = ""
        for i in range(len(results)):
            chunk = results[i]
            source = chunk.metadata["source"]
            retval = retval + f"From http://{source}\n\n"
            retval = retval + chunk.page_content
            retval = retval + "\n====================\n\n"

        return retval

    async def __call__(self, request: Request) -> list[str]:
        return self.search(request.query_params["query"])

deployment = VectorSearchDeployment.bind()
# serve.run(deployment)

if __name__ == "__main__":
    embs = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    db = FAISS.load_local(FAISS_INDEX_PATH, embs, allow_dangerous_deserialization=True)
