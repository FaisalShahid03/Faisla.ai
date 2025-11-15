import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT, ID
import os

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # reranker

# ---- Step 2: Load saved FAISS index + mappings ----
index_faiss = faiss.read_index("legal_index.faiss")

with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

with open("file_map.pkl", "rb") as f:
    file_to_chunk = pickle.load(f)

# ---- Step 3: Load or build Whoosh index ----
if not os.path.exists("whoosh_index"):
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    os.mkdir("whoosh_index")
    ix = index.create_in("whoosh_index", schema)
    writer = ix.writer()
    for i, chunk in enumerate(all_chunks):
        writer.add_document(id=str(i), content=chunk)
    writer.commit()
else:
    ix = index.open_dir("whoosh_index")

# ---- Step 4: Hybrid Retrieval ----
def hybrid_retrieve(query, top_k=20):
    # FAISS search
    query_vec = bi_encoder.encode([query], convert_to_numpy=True)
    distances, indices = index_faiss.search(query_vec, k=top_k)
    faiss_results = [(idx, 1 - float(dist)) for idx, dist in zip(indices[0], distances[0])]

    # BM25 search
    with ix.searcher() as searcher:
        q = QueryParser("content", ix.schema).parse(query)
        bm25_hits = searcher.search(q, limit=top_k)
        bm25_results = [(int(hit["id"]), hit.score) for hit in bm25_hits]

    # Merge scores
    scores = {}
    for idx, score in faiss_results:
        scores[idx] = scores.get(idx, 0) + 0.5 * score
    for idx, score in bm25_results:
        scores[idx] = scores.get(idx, 0) + 0.5 * score

    # Sort by score
    candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return candidates

# ---- Step 5: Reranking ----
def rerank(query, candidates, top_k=5):
    pairs = [(query, all_chunks[idx]) for idx, _ in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, ((idx, base_score), rerank_score) in enumerate(ranked, start=1):
        results.append({
            "rank": rank,
            "file": file_to_chunk[idx],
            "base_score": base_score,
            "rerank_score": float(rerank_score),
            "text": all_chunks[idx][:400] + "..."
        })
    return results

# ---- Step 6: Interactive Loop ----
while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    candidates = hybrid_retrieve(query, top_k=20)
    results = rerank(query, candidates, top_k=3)

    print("\n🔍 Final Reranked Results:")
    for r in results:
        print(f"\nRank {r['rank']} | File: {r['file']} | Rerank Score: {r['rerank_score']:.4f}")
        print(r['text'])
