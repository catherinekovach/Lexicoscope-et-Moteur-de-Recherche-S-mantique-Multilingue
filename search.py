import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# files created from build_index.py
INDEX_FILENAME = "combined_corpus.index"
TEXT_FILENAME = "combined_sentences.pkl"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


    #Displays the Key Word In Context (KWIC)
    #Shows the matched sentence along with the sentence before and after it.
def display_kwic(result_indices, distances, all_sentences, window=1):
    
    for i, idx in enumerate(result_indices[0]):
        score = distances[0][i]
        print(f"\n" + "-"*50)
        print(f"Match {i+1} | Similarity Score: {score:.4f}")
        print("-" * 50)
        
        # Calculate the window range 
        start_idx = max(0, idx - window)
        end_idx = min(len(all_sentences), idx + window + 1)
        
        for current_pos in range(start_idx, end_idx):
            sentence = all_sentences[current_pos]
            
            # Highlight the actual match with arrows
            if current_pos == idx:
                print(f"  >>> {sentence}")
            else:
                print(f"      {sentence}")

#Search engine

print("=> Starting search engine <=")


print(f"1. Loading Model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

# Load the Faiss Index 
print(f"2. Loading Faiss Index from '{INDEX_FILENAME}'...")
index = faiss.read_index(INDEX_FILENAME)

# Loading the text
print(f"3. Loading Text Corpus from '{TEXT_FILENAME}'...")
with open(TEXT_FILENAME, "rb") as f:
    sentences = pickle.load(f)

print(f"\nSUCCESS! Loaded {index.ntotal} sentences into memory.")
print("\n" + "=>  MULTILINGUAL SEMANTIC CONCORDANCER <=")

# Step 4: The Interactive Loop
while True:
    query = input("\nEnter a search query (or type 'exit' to quit): \n> ")
    
    if query.strip().lower() == 'exit':
        print("Shutting down Concordancer. Goodbye, friend!")
        break
    
    if not query.strip():
        continue
        
    # Encode the user's query
    query_vector = model.encode([query], convert_to_numpy=True)
    
    # Normalize the query vector for Cosine Similarity
    faiss.normalize_L2(query_vector)
    
    # Search the Faiss Index
    k = 5  # Number of top results you want to retrieve
    distances, indices = index.search(query_vector, k)
    
    # Display the results with context
    display_kwic(indices, distances, sentences, window=1)
