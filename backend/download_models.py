import os
import sys
from huggingface_hub import snapshot_download

def download_huggingface_models():
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        print("CRITICAL BUILD ERROR: HF_REPO_ID environment variable is missing!")
        sys.exit(1)

    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    print(f"Downloading models from Hugging Face repo: {repo_id} to {dest_dir}...")
    hf_token = os.environ.get("HF_TOKEN") 
    if not hf_token:
        print("WARNING: No HF_TOKEN detected in environment variables. If this repo is private, it will fail.")
        
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
