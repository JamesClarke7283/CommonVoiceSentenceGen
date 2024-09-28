import logging
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

log_file_path = os.path.join(os.path.dirname(__file__), '..', 'gen-common-voice.log')

logger = logging.getLogger('gen_common_voice')
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to logger
if not logger.handlers:
    logger.addHandler(fh)

# Prevent logger from propagating to the root logger
logger.propagate = False
