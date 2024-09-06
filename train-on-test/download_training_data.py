import requests
import os

def download_files(paths, download_directory):
    """
    Download files from the given list of URLs into the specified directory.

    :param paths: List of file URLs.
    :param download_directory: Path to the directory where files will be saved.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    def download_file(url, directory):
        try:
            # Get the file name from the URL
            filename = os.path.join(directory, url.split("/")[-1])
            
            # Stream the file download to avoid memory issues with large files
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                        file.write(chunk)
            print(f"Successfully downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")

    # Download each file from the list
    for url in paths:
        download_file(url, download_directory)


    print("Download completed.")
if __name__ == "__main__":
    # specify the config file as a command line argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", default="../configs/official/OLMo-1B.yaml")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    from olmo.config import TrainConfig
    cfg = TrainConfig.load(args.train_config_path)
    training_files = cfg.data.paths

    # wand logging
    import wandb
    os.environ["WANDB__SERVICE_WAIT"]="6000"
    wandb.init(
        name="download_training_data",
        project="olmo-small",
    )

    # download all the files, curtesy to the function above written by chatgpt
    download_files(training_files, args.output_dir)