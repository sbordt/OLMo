import json
import os
import sqlite3
import threading
from datasets import Dataset
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry
import numpy as np
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.tokenizer import Tokenizer


# Thread-local storage for SQLite connections
local = threading.local()

def get_db_connection(db_path):
    """Get a thread-local database connection."""
    if not hasattr(local, 'conn'):
        local.conn = sqlite3.connect(db_path)
        # Create the chunks table if it doesn't exist
        local.conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        local.conn.commit()
    return local.conn


@retry(wait='exponential', stop=(10, 60))
def download_chunk(url, start_byte, end_byte):
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 206:  # 206 indicates a successful partial content request
        return response.content
    else:
        raise ValueError(f"Failed to download chunk from {url} with status code {response.status_code}")


def download_dataset_chunk(dataset, url:str, index:int, db_path:str, tokenizer):
    """Download a single chunk and save it to database immediately"""
    # Create a unique ID for this chunk
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    chunk_id = f"{url_hash}_{index}"
    
    # Get database connection (thread-local)
    conn = get_db_connection(db_path)
    
    # Check if this chunk has already been downloaded
    cursor = conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,))
    result = cursor.fetchone()
    if result:
        print(f"Chunk {chunk_id} already exists in database, skipping download")
        return result[0]
    
    # If not, proceed with download
    dtype = dataset.dtype
    item_size = dtype(0).itemsize
    bytes_start = index * item_size * dataset._chunk_size
    num_bytes = item_size * dataset._chunk_size
    
    print(f"Downloading chunk {index}...")
    batch_bytes = download_chunk(url, bytes_start, bytes_start+num_bytes-1)
    tokens = np.frombuffer(batch_bytes, dtype=dataset.dtype).tolist()
    
    # Decode tokens to text
    text = tokenizer.decode(tokens)
    
    # Save the decoded text to database immediately
    try:
        conn.execute(
            "INSERT INTO chunks (id, url, chunk_index, content) VALUES (?, ?, ?, ?)",
            (chunk_id, url, index, text)
        )
        conn.commit()
        print(f"Saved chunk {chunk_id} to database")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # If there was an error, try to rollback
        conn.rollback()
    
    return text


def download_dataset_chunks_simultaneously(dataset, metadata, db_path, tokenizer, max_workers=48):
    """Asynchronously download different sequences in the batch, but keep the sequence order."""
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor and store the future and its corresponding index in a dictionary
        for i, x in enumerate(metadata):
            future = executor.submit(download_dataset_chunk, dataset, x[0], x[1], db_path, tokenizer)
            futures[future] = i

        # Create a results list of the same size as the number of futures
        results = [None] * len(futures)
        
        # Iterate over futures as they complete
        for future in as_completed(futures):
            index = futures[future]  # Retrieve the original index for this future
            try:
                results[index] = future.result()  # Store result at the correct index
            except Exception as e:
                print(f"Error downloading chunk at index {index}: {e}")
    
    return results 


def load_chunks_from_db(db_path, metadata=None):
    """Load all chunks from database, optionally filtering by metadata"""
    conn = sqlite3.connect(db_path)
    texts = []
    
    if metadata:
        # Load specific chunks based on metadata
        for url, index in metadata:
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()
            chunk_id = f"{url_hash}_{index}"
            
            cursor = conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,))
            result = cursor.fetchone()
            if result:
                texts.append(result[0])
            else:
                print(f"Warning: Chunk {chunk_id} not found in database")
    else:
        # Load all chunks in order of chunk_index
        cursor = conn.execute("SELECT content FROM chunks ORDER BY chunk_index")
        for row in cursor.fetchall():
            texts.append(row[0])
    
    conn.close()
    return texts


def process_metadata_to_hf_dataset(metadata_file, db_path, dataset_dir, args):
    """
    Process a JSON metadata file, download chunks to a SQLite database,
    and convert it to a HuggingFace dataset.
    
    Args:
        metadata_file: Path to the JSON metadata file
        db_path: Path to the SQLite database file
        dataset_dir: Directory to save the HuggingFace dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load metadata from JSON file
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load the dataset configuration
    cfg = TrainConfig.load(args.config)
    dataset = build_memmap_dataset(cfg, cfg.data)
    
    # Load the tokenizer for decoding
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Download chunks in parallel, saving each to database as they are received
    download_dataset_chunks_simultaneously(dataset, metadata, db_path, tokenizer, max_workers=args.workers)
    
    # Load all chunks from database
    print("Loading all chunks from database to create dataset...")
    texts = load_chunks_from_db(db_path, metadata)
    
    # Create the dataset from the loaded texts
    combined_data = [{"text": text} for text in texts if text]
    
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list(combined_data)
    
    # Save the dataset to disk
    hf_dataset.save_to_disk(dataset_dir)
    print(f"Dataset saved to {dataset_dir}")
    
    # Optionally, update the database to mark these chunks as processed
    conn = sqlite3.connect(db_path)
    for url, index in metadata:
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        chunk_id = f"{url_hash}_{index}"
        conn.execute("UPDATE chunks SET processed = 1 WHERE id = ?", (chunk_id,))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Convert metadata JSON to HuggingFace dataset")
    parser.add_argument("--metadata", help="Path to metadata JSON file", default="mid_training_metadata.json")
    parser.add_argument("--db-path", help="Path to SQLite database file", default="chunks.db")
    parser.add_argument("--dataset-dir", help="Directory to save the final HuggingFace dataset", default="mid_training_dataset")
    parser.add_argument("--workers", type=int, default=48, help="Maximum number of workers for parallel processing")
    parser.add_argument("--tokenizer", default="../olmo_data/tokenizers/allenai_dolma2.json", help="this is for Olmo2. Olmo1 uses a different tokenizer")
    parser.add_argument("--config", default="../configs/official-1124/OLMo2-7B-stage2-seed42.yaml", help="Path to training config file")
    
    args = parser.parse_args()
    
    process_metadata_to_hf_dataset(args.metadata, args.db_path, args.dataset_dir, args)


if __name__ == "__main__":
    main()