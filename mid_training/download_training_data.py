import json
import os
from datasets import Dataset
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry
import numpy as np
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.tokenizer import Tokenizer


@retry(wait='exponential', stop=(10, 60))
def download_chunk(url, start_byte, end_byte):
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 206:  # 206 indicates a successful partial content request
        return response.content
    else:
        raise ValueError(f"Failed to download chunk from {url} with status code {response.status_code}")


def download_dataset_chunk(dataset, url:str, index :int):
    dtype = dataset.dtype
    item_size = dtype(0).itemsize
    bytes_start = index * item_size * dataset._chunk_size
    num_bytes = item_size * dataset._chunk_size
    batch_bytes = download_chunk(url, bytes_start, bytes_start+num_bytes-1)
    return np.frombuffer(batch_bytes, dtype=dataset.dtype).tolist()


def download_dataset_chunks_simultaneously(dataset, metadata, max_workers=48):
    """Asynchroniosly download different sequences in the batch, but keep the sequence order. Courtesy of ChatGPT."""
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor and store the future and its corresponding index in a dictionary
        for i, x in enumerate(metadata):
            future = executor.submit(download_dataset_chunk, dataset, x[0], x[1])
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


def process_metadata_to_hf_dataset(metadata_file, output_dir, args):
    """
    Process a JSON metadata file and convert it to a HuggingFace dataset.
    
    Args:
        metadata_file: Path to the JSON metadata file
        output_dir: Directory to save the HuggingFace dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata from JSON file
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Assuming metadata is a list of [start, end] pairs
    # You might need to adjust this based on your actual metadata structure
    
    # This is a placeholder for your actual dataset
    # Replace with your actual dataset loading logic
    # For example: dataset = load_dataset("your_dataset")
    cfg = TrainConfig.load(args.config)
    dataset = build_memmap_dataset(cfg, cfg.data)
    
    # Download chunks in parallel
    results = download_dataset_chunks_simultaneously(dataset, metadata)
    
    # Combine results into a single dataset
    # This will depend on your specific data structure
    # Here's a simple example assuming results are compatible with HF datasets
    tokenizer = Tokenizer.from_file(args.tokenizer)

    combined_data = []
    for tokens in results:
        if tokens is None:
            continue
        text = tokenizer.decode(tokens)
        combined_data.append({"text": text})

    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list(combined_data)
    
    # Save the dataset to disk
    hf_dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert metadata JSON to HuggingFace dataset")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON file")
    parser.add_argument("--output", required=True, help="Output directory for HuggingFace dataset")
    parser.add_argument("--workers", type=int, default=48, help="Maximum number of workers for parallel processing")
    parser.add_argument("--tokenizer", default="../olmo_data/tokenizers/allenai_dolma2.json", help="this is for Olmo2. Olmo1 uses a different tokenizer")
    parser.add_argument("--config", default="../configs/official-1124/OLMo2-7B-stage2-seed42.yaml", help="Path to training config file")
    
    args = parser.parse_args()
    
    process_metadata_to_hf_dataset(args.metadata, args.output, args)

if __name__ == "__main__":
    main()