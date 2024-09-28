import os
import tomllib  # For Python 3.11 and above
import hashlib

# For Python < 3.11, use 'tomli' instead:
# import tomli as tomllib

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
DB_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'gen-common-voice-data.db')

def load_config():
    with open(CONFIG_FILE_PATH, 'rb') as f:
        config = tomllib.load(f)
    return config

def get_config_hash():
    with open(CONFIG_FILE_PATH, 'rb') as f:
        file_content = f.read()
        return hashlib.sha256(file_content).hexdigest()
