import os
import shutil

# Paths (ensure these paths are correct and adjust if necessary)
sources_dir = '../sources/'
embeddings_dir = '../faiss/'
processed_files_log = '../faiss/processed_files.json'
faiss_index_path = '../faiss/faiss_index.bin'
faiss_metadata_path = '../faiss/faiss_metadata.json'

def reset_faiss_database():
    """Resets FAISS index, embeddings, metadata, and processed files."""
    
    # Step 1: Delete the FAISS index file
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
        print(f"Deleted FAISS index at {faiss_index_path}")
    else:
        print(f"FAISS index not found at {faiss_index_path}")
    
    # Step 2: Delete the FAISS metadata file
    if os.path.exists(faiss_metadata_path):
        os.remove(faiss_metadata_path)
        print(f"Deleted FAISS metadata at {faiss_metadata_path}")
    else:
        print(f"FAISS metadata not found at {faiss_metadata_path}")
    
    # Step 3: Delete all embeddings files
    if os.path.exists(embeddings_dir):
        for filename in os.listdir(embeddings_dir):
            if filename.endswith('_embeddings.npy'):
                file_path = os.path.join(embeddings_dir, filename)
                os.remove(file_path)
                print(f"Deleted embeddings file: {file_path}")
    else:
        print(f"Embeddings directory not found at {embeddings_dir}")
    
    # Step 4: Delete the processed files log
    if os.path.exists(processed_files_log):
        os.remove(processed_files_log)
        print(f"Deleted processed files log at {processed_files_log}")
    else:
        print(f"Processed files log not found at {processed_files_log}")
    
    # Optional: Delete the chunked text files if needed
    for filename in os.listdir(sources_dir):
        if filename.endswith('_chunks.txt'):
            file_path = os.path.join(sources_dir, filename)
            os.remove(file_path)
            print(f"Deleted chunked text file: {file_path}")
    
    print("FAISS database reset complete.")

if __name__ == '__main__':
    reset_faiss_database()
