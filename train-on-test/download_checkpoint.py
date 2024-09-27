import requests
import os

# Define a function to download a file from a URL
def download_file(url, directory, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} to {filepath}")
    else:
        print(f"Failed to download {filename} from {url}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_url", default="https://olmo-checkpoints.org/ai2-llm/olmo-medium/6qvoqf3c/step300000-unsharded")
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
