import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))
app_dir = os.path.dirname(script_dir)

# Construct absolute paths based on the app's directory
sources_dir = os.path.join(app_dir, 'sources/')
embeddings_dir = os.path.join(app_dir, 'faiss/')
processed_files_log = os.path.join(app_dir, 'faiss', 'processed_files.json')
faiss_index_path = os.path.join(app_dir, 'faiss', 'faiss_index.bin')
faiss_metadata_path = os.path.join(app_dir, 'faiss', 'faiss_metadata.json')

# ========== Step 1: Process New Files ==========
def load_processed_files():
    """Load the list of processed files."""
    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r') as file:
            return json.load(file)
    return []

def save_processed_files(processed_files):
    """Save the list of processed files."""
    with open(processed_files_log, 'w') as file:
        json.dump(processed_files, file, indent=2)

def preprocess_and_chunk(text, max_length=4000):  # Increase chunk size from 1000 to 4000 tokens
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_new_files():
    """Process newly added files in the sources directory."""
    processed_files = load_processed_files()
    
    # Get all text files in the sources directory
    files_to_process = [f for f in os.listdir(sources_dir) if f.endswith('.txt') and f not in processed_files]
    
    if not files_to_process:
        print("No new files to process.")
        return []
    
    for filename in files_to_process:
        file_path = os.path.join(sources_dir, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Chunk the text
        chunks = preprocess_and_chunk(text)
        print(f"Total chunks created from {filename}: {len(chunks)}")

        # Save the chunks for embedding processing
        chunks_file_path = os.path.join(sources_dir, f"{filename}_chunks.txt")
        with open(chunks_file_path, 'w') as chunk_file:
            for chunk in chunks:
                chunk_file.write(chunk + '\n\n')
        
        # Mark this file as processed
        processed_files.append(filename)
    
    save_processed_files(processed_files)
    return files_to_process

# ========== Step 2: Generate Embeddings ==========
def load_chunks(file_path):
    """Load text chunks from a file."""
    with open(file_path, 'r') as file:
        return [chunk.strip() for chunk in file.read().split('\n\n') if chunk.strip()]

def generate_embeddings(chunks, batch_size=16, model_name='multi-qa-mpnet-base-dot-v1'):
    """Generate embeddings for the given text chunks with batching."""
    embedding_model = SentenceTransformer(model_name)
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings).astype('float32')


def process_chunks_for_embeddings(files_to_process):
    """Generate embeddings for the processed files."""
    for file in files_to_process:
        chunks_file_path = os.path.join(sources_dir, f"{file}_chunks.txt")
        embeddings_file_path = os.path.join(embeddings_dir, f"{file}_embeddings.npy")
        
        # Load the text chunks
        chunks = load_chunks(chunks_file_path)
        if not chunks:
            print(f"No chunks found in {chunks_file_path}")
            continue
        
        # Generate embeddings for the chunks
        embeddings = generate_embeddings(chunks)
        print(f"Generated embeddings for {len(chunks)} chunks from {file}.")
        
        # Save embeddings to a file
        np.save(embeddings_file_path, embeddings)
        print(f"Embeddings saved to {embeddings_file_path}")

# ========== Step 3: Update FAISS Index and Metadata ==========
def load_embeddings(file_path):
    """Load embeddings from a file."""
    return np.load(file_path)

def update_faiss_index(embeddings, index_path):
    """Load an existing FAISS index and add new embeddings to it."""
    if not os.path.exists(index_path):
        print(f"FAISS index not found at {index_path}. Initializing a new one.")
        d = embeddings.shape[1]  # Dimensionality of the embeddings
        index = faiss.IndexHNSWFlat(d, 32)  # HNSW with 32 neighbors
        
        # Optional: Set parameters for higher recall or performance
        index.hnsw.efSearch = 64  # Default is 16, increase for better recall
        index.hnsw.efConstruction = 40  # Default is 40, adjust if needed
    else:
        index = faiss.read_index(index_path)
    
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Updated FAISS index with {len(embeddings)} new embeddings.")


def update_metadata(chunks, metadata_path, source):
    """Update the FAISS metadata with new chunk information."""
    if not os.path.exists(metadata_path):
        metadata = []
    else:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

    current_length = len(metadata)
    new_metadata = []
    
    for i, chunk in enumerate(chunks):
        entry = {
            "id": current_length + i,
            "text": chunk,
            "source": source,
            "chunk": i + 1
        }
        new_metadata.append(entry)
    
    metadata.extend(new_metadata)
    
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=2)
    print(f"Updated metadata with {len(new_metadata)} new entries.")

def update_faiss_and_metadata(files_to_process):
    """Update FAISS index and metadata for the processed embeddings."""
    for file in files_to_process:
        embeddings_file_path = os.path.join(embeddings_dir, f"{file}_embeddings.npy")
        chunks_file_path = os.path.join(sources_dir, f"{file}_chunks.txt")
        
        # Load embeddings and chunks
        embeddings = load_embeddings(embeddings_file_path)
        chunks = load_chunks(chunks_file_path)
        
        # Update FAISS index
        update_faiss_index(embeddings, faiss_index_path)
        
        # Update metadata
        update_metadata(chunks, faiss_metadata_path, source=file)

# ========== Main Script ==========
if __name__ == '__main__':
    # Step 1: Process new files
    files_to_process = process_new_files()
    
    # If there are no new files to process, exit
    if not files_to_process:
        print("No new files to process. Exiting.")
    else:
        # Step 2: Generate embeddings for the processed chunks
        process_chunks_for_embeddings(files_to_process)
        
        # Step 3: Update the FAISS index and metadata
        update_faiss_and_metadata(files_to_process)

    print("Processing complete.")
