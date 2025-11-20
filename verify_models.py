import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.services.vad_iterator import VADIterator
from app.services.acoustic_analyzer import AcousticAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_downloads():
    logger.info("Verifying VAD Model Download...")
    try:
        vad = VADIterator()
        logger.info("VAD Model Verified.")
    except Exception as e:
        logger.error(f"VAD Verification Failed: {e}")

    logger.info("Verifying Acoustic Model Download...")
    try:
        acoustic = AcousticAnalyzer()
        logger.info("Acoustic Model Verified.")
    except Exception as e:
        logger.error(f"Acoustic Verification Failed: {e}")

if __name__ == "__main__":
    verify_downloads()
