import time

import numpy as np
import ray
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embeddings import LocalHuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index_fast"
db_shards = 8

loader = ReadTheDocsLoader("docs.ray.io/en/master/")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)


@ray.remote(num_cpus=2)
def process_shard(shard):
    print(f"Processing shard of {len(shard)} chunks...")
    st = time.time()
    embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    result = FAISS.from_documents(shard, embeddings)
    et = time.time() - st
    print(f"Shard indexed in {et:.2f} seconds")
    return result


st = time.time()
print("Loading documents...")

docs = loader.load()

chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
)

et = time.time() - st
print(f"Time taken to split documents: {et:.2f} seconds in {len(chunks)} chunks")

print(f"Loading chunks into vector store... using {db_shards} shards")
st = time.time()
shards = np.array_split(chunks, db_shards)
futures = [process_shard.remote(shards[i]) for i in range(db_shards)]
results = ray.get(futures)
et = time.time() - st
print(f"Shard processing completed. Time taken: {et:.2f} seconds")

st = time.time()
print("Merging shards...")
db = results[0]
for i in range(1, db_shards):
    db.merge_from(results[i])
et = time.time() - st
print(f"Shards merged. Time taken: {et:.2f} seconds")

st = time.time()
print("Saving index...")
db.save_local(FAISS_INDEX_PATH)
et = time.time() - st
print(f"Index saved. Time taken: {et:.2f} seconds")
