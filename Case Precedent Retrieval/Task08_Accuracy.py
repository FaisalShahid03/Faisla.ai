import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT, ID
import os

# ---- Step 1: Load models ----
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # embeddings
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
            "text": all_chunks[idx]
        })
    return results


# ---- Step 6: Query List ----
queries = [
    "What are the legal grounds for self-defence under Australian law?",
    "Explain how negligence is established in Australian tort law.",
    "What is the difference between manslaughter and murder in Australia?",
    "Under what circumstances can a contract be considered void in Australia?",
    "What are the tenant’s rights in residential lease agreements in Australia?",
    "How does Australian law define defamation?",
    "What are the penalties for insider trading in Australia?",
    "Can a minor be held liable for breach of contract under Australian law?",
    "What is the process of judicial review in Australian administrative law?",
    "What constitutes unfair dismissal under Australian employment law?",
    "Explain the rules around search and seizure under Australian criminal law.",
    "What are the requirements for granting bail in Australia?",
    "Describe the legal concept of duty of care in Australian negligence cases.",
    "What is the role of precedent in the Australian legal system?",
    "How are damages calculated in Australian personal injury cases?",
    "Can a company be found guilty of a criminal offence in Australia?",
    "What are the principles of equity recognized by Australian courts?",
    "How does the Australian High Court handle constitutional disputes?",
    "Explain how family law handles child custody in Australia.",
    "What remedies are available for breach of contract under Australian law?"
]

# ---- Step 7: Automated Retrieval + Save to File ----
output_file = "retrieval_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, query in enumerate(queries, start=1):
        f.write("=" * 80 + "\n")
        f.write(f"Query {i}: {query}\n")
        f.write("=" * 80 + "\n")

        candidates = hybrid_retrieve(query, top_k=20)
        results = rerank(query, candidates, top_k=3)

        for r in results:
            f.write(f"\nRank {r['rank']} | File: {r['file']} | "
                    f"Base Score: {r['base_score']:.4f} | Rerank Score: {r['rerank_score']:.4f}\n")
            f.write(r['text'] + "\n")

        f.write("\n\n")

print(f"\n✅ Retrieval complete! Results saved to '{output_file}'")
