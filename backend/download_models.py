import os
import sys
from huggingface_hub import snapshot_download

def download_huggingface_models():
    # Read the repository ID from an environment variable (set this in Render)
    # Example: "Kaviyathamizhan/sentiment-models"
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        print("HF_REPO_ID environment variable is not set. Skipping model download.")
        return

    # Download to the root /models directory or inside backend based on where predict.py looks for it
    # predict.py uses OS path up two directories, so the root repository folder
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    print(f"Downloading models from Hugging Face repo: {repo_id}...")
    try:
        # If the repository is private, Render needs to have HF_TOKEN environment variable set
        hf_token = os.environ.get("HF_TOKEN") 
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="model", # or "dataset" depending on how they uploaded
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print("Models successfully downloaded!")
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_huggingface_models()
