import os
import requests
from tqdm import tqdm  # for displaying a progress bar

def download_file(url, directory, filename, chunk_size=1024*1024):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        total_size = int(response.headers.get('content-length', 0))
        progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    progress.update(len(chunk))

        progress.close()

        if total_size != 0 and progress.n != total_size:
            print(f"WARNING: Downloaded file size does not match expected size for {filename}.")
        else:
            print(f"Successfully downloaded {filename} to {filepath}")
    else:
        print(f"Failed to download {filename} from {url}, status code: {response.status_code}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_url", default="https://olmo-checkpoints.org/ai2-llm/olmo-medium/6qvoqf3c/step300000-unsharded") # OLMo-7B-0424
    parser.add_argument("--output_dir", default="/mnt/qb/luxburg/sbordt10/OLMo-7B-checkpoints/step300000-unsharded")
    args = parser.parse_args()

    # wand logging
    import wandb
    os.environ["WANDB__SERVICE_WAIT"]="6000"
    wandb.init(
        name="download_checkpoint",
        project="olmo-small",
    )
        
    # the files to download
    files = ['config.yaml', 'model.pt', 'optim.pt', 'train.pt']
        
    # Download each file
    for file in files:
        download_file(f"{args.checkpoint_url}/{file}", args.output_dir, file)
