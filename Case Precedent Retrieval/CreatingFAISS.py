import os
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ---- Step 1: Load embedding model ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Step 2: Chunking function ----
def chunk_sentences(sentences, max_words=250):
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---- Step 3: Prepare FAISS index ----
dimension = 384  # embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

all_chunks = []  # to map back results
file_to_chunk = []  # keeps track of which file each chunk came from

# ---- Step 4: Loop through all files in data/ ----
data_folder = "data"
for fname in os.listdir(data_folder):
    if fname.endswith(".xml"):
        file_path = os.path.join(data_folder, fname)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            soup = BeautifulSoup(raw_text, "xml")

            # Extract sentences
            sentences = [s.get_text(strip=True) for s in soup.find_all("sentence")]
            if not sentences:
                continue

            # Chunk sentences
            chunks = chunk_sentences(sentences)

            # Encode embeddings
            embeddings = model.encode(chunks, convert_to_numpy=True)

            # Add to FAISS
            index.add(embeddings)

            # Save mapping info
            all_chunks.extend(chunks)
            file_to_chunk.extend([fname] * len(chunks))

            print(f"Processed {fname}, chunks: {len(chunks)}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

print("\n✅ Finished indexing all files!")
print("Total chunks stored in FAISS:", index.ntotal)

# ---- Step 5: Save index & mapping ----
faiss.write_index(index, "legal_index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

with open("file_map.pkl", "wb") as f:
    pickle.dump(file_to_chunk, f)

print("Index and mappings saved!")

# ---- Step 6: Quick test query ----
query = "copyright law in Australia"
query_vec = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_vec, k=3)

print("\n🔍 Top 3 Retrieved Chunks for Query:")
for i, idx in enumerate(indices[0]):
    print(f"\nResult {i+1} (distance={distances[0][i]:.4f}) from {file_to_chunk[idx]}:")
    print(all_chunks[idx][:400], "...")
