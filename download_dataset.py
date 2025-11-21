import kagglehub
import shutil
import os
import glob

def download_ravdess():
    print("Downloading RAVDESS dataset via KaggleHub...")
    try:
        # Download to cache
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        print(f"Dataset downloaded to cache: {path}")
        
        # Define local target directory
        target_dir = "ravdess_data"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        print(f"Copying files to local directory: {target_dir}...")
        
        # Copy audio files
        wav_files = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
        count = 0
        for file in wav_files:
            # Keep folder structure or flatten? Flattening is easier for this script
            # But RAVDESS has Actor_XX folders. Let's keep structure if possible, 
            # or just copy all to one folder for simplicity in benchmarking.
            # Let's flatten for simplicity as filenames contain all info.
            filename = os.path.basename(file)
            shutil.copy2(file, os.path.join(target_dir, filename))
            count += 1
            
        print(f"Successfully copied {count} files to {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_ravdess()
