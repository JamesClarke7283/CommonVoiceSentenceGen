import os
from tortoise import models, fields, Tortoise
from tortoise.fields import JSONField

DB_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'gen-common-voice-data.db')
DB_URL = f'sqlite://{DB_FILE_PATH}'

class ConfigModel(models.Model):
    id = fields.IntField(pk=True)
    config_hash = fields.CharField(max_length=64)

class SentenceModel(models.Model):
    id = fields.IntField(pk=True)
    text = fields.TextField()
    is_validated = fields.BooleanField(default=False)
    is_final_validated = fields.BooleanField(default=False)
    domains = JSONField(null=True)  # Stores list of domains
    score = fields.IntField(null=True)

async def init_db():
    await Tortoise.init(
        db_url=DB_URL,
        modules={'models': ['src.models']}
    )
    await Tortoise.generate_schemas()
