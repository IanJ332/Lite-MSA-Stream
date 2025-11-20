import os
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

def download_model(repo_id: str, filename: str, local_dir: str = "models") -> str:
    """
    Downloads a model from Hugging Face Hub to a local directory.
    
    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        filename (str): The filename of the model in the repository.
        local_dir (str): The local directory to save the model to.
        
    Returns:
        str: The absolute path to the downloaded model file.
    """
    try:
        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        logger.info(f"Checking model {filename} from {repo_id}...")
        
        # Download or get from cache
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Ensure we get an actual file
        )
        
        logger.info(f"Model available at: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model {filename}: {e}")
        raise RuntimeError(f"Could not download model: {e}")
