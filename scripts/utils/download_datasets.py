import kagglehub
import shutil
import os

def move_dataset(src_path, dest_folder):
    """Moves dataset files from Kaggle cache to project folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created {dest_folder}")
    
    print(f"Moving files from {src_path} to {dest_folder}...")
    
    # Walk through source and move files
    for root, dirs, files in os.walk(src_path):
        for file in files:
            src_file = os.path.join(root, file)
            
            # Maintain subdirectory structure if needed, or just flatten?
            # TESS has subfolders (OAF_angry, etc.), RAVDESS has Actor_01, etc.
            # Let's keep structure relative to src_path
            rel_path = os.path.relpath(root, src_path)
            dest_dir = os.path.join(dest_folder, rel_path)
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            dest_file = os.path.join(dest_dir, file)
            
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file) # Copy instead of move to be safe with cache
                
    print(f"Successfully populated {dest_folder}")

def main():
    print("--- Downloading Datasets via KaggleHub ---")

    # 1. CREMA-D
    print("\n[1/3] Downloading CREMA-D...")
    try:
        path = kagglehub.dataset_download("ejlok1/cremad")
        print("Kaggle Path:", path)
        move_dataset(path, "crema_data")
    except Exception as e:
        print(f"Failed to download CREMA-D: {e}")

    # 2. TESS
    print("\n[2/3] Downloading TESS...")
    try:
        path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
        print("Kaggle Path:", path)
        move_dataset(path, "tess_data")
    except Exception as e:
        print(f"Failed to download TESS: {e}")

    # 3. RAVDESS (Ensure we have the full version)
    print("\n[3/3] Downloading RAVDESS...")
    try:
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        print("Kaggle Path:", path)
        move_dataset(path, "ravdess_data")
    except Exception as e:
        print(f"Failed to download RAVDESS: {e}")

    print("\n--- Download Complete ---")
    print("Datasets are ready in:")
    print("- crema_data/")
    print("- tess_data/")
    print("- ravdess_data/")

if __name__ == "__main__":
    main()
