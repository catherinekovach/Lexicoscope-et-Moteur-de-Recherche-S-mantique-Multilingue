import os
import re
import faiss
import nltk
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Enter folder path
FOLDER_PATH = r"C:\Users\catch\Documents\M1s2 IDL\Python\Project\Guardian_lemonde_corpus"

# Huggingface 
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Filenames for saved data
INDEX_FILENAME = "guardian_corpus.index"
TEXT_FILENAME = "guardian_sentences.pkl"

def clean_and_extract_sentences(folder_path):
    """Reads all .txt files in a folder, cleans XML tags, tags source, and extracts sentences."""
    all_sentences = []
    
    # Get a list of all .txt files in the directory
    #Only applicable if you have multiple .txt files
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"Found {len(valid_files)} text files in the directory.")
    
    for filename in valid_files:
        filepath = os.path.join(folder_path, filename)
        
        # Determine the source tag based on the filename
        # Helpful with multiple corpora
        if 'monde' in filename.lower():
            source_tag = "[LE MONDE] "
        else:
            source_tag = "[GUARDIAN] "
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
            # This removes all XML tags like <doc id="123"> or </p>
            # Just in case there was anything left over from the transfer from .txt
            text_no_xml = re.sub(r'<[^>]+>', ' ', text)
            
            # Split into sentences
            raw_sentences = nltk.sent_tokenize(text_no_xml)
            
            for s in raw_sentences:
                cleaned = s.strip().replace('\n', ' ')
                # Remove multiple spaces caused by stripping XML
                cleaned = re.sub(r'\s+', ' ', cleaned) 
                
                if len(cleaned) > 15: # Filter out tiny fragments
                    # Add the newspaper tag to the front of the sentence!
                    final_sentence = source_tag + cleaned
                    all_sentences.append(final_sentence)
                    
    return all_sentences

#Multiple printed steps to discover location of the bug

print("=> INDEX BUILDER <=")

# read all files
print(f"\nScanning folder: {FOLDER_PATH}")
sentences = clean_and_extract_sentences(FOLDER_PATH)

#Sentences cleaned
print(f"Extracted {len(sentences)} clean sentences :)")

# Load Model
print(f"\nLoading AI Model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

# Encode (This is dependant to how many files you have)
# Change batch size if you run into a problem here
print("Encoding sentences into vectors.")
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# Normalize and build Faiss
# Look at website to understand normalize_L2 function
print("\nNormalizing vectors and building Faiss index.")
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings)


print("\nSaving files.")

# Save the Faiss index 
faiss.write_index(index, INDEX_FILENAME)
print(f"Saved: {INDEX_FILENAME}")

# Save the sentences from text
with open(TEXT_FILENAME, "wb") as f:
    pickle.dump(sentences, f)
print(f"Saved: {TEXT_FILENAME}")

print("\n=> BUILD COMPLETE <=")
