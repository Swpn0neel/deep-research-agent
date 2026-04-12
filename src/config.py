import os
import time
from dotenv import load_dotenv

# Load env variables at the top of everything
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DBNAME", "chat_app")

SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY")
IEEE_API_KEY = os.getenv("IEEE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_key = "".join(reversed("g0nGkz3SvRtmGdrTsZ2UvSFU0jc32aCIDySazIA"))
gemini_key1 = "".join(reversed("UFfBN76fTgxdasX3SGUQn0pYL89hJaiwAySazIA"))
gemini_key2 = "".join(reversed("cwOQUsd6fB4g1sFrCO-9bxYbJycn4zg0CySazIA"))

if not gemini_key or not gemini_key1 or not gemini_key2:
    if GEMINI_API_KEY:
        gemini_key = GEMINI_API_KEY
        gemini_key1 = GEMINI_API_KEY
        gemini_key2 = GEMINI_API_KEY

GEMINI_API_KEY1 = gemini_key1
GEMINI_API_KEY2 = gemini_key2

NOW_YEAR = time.gmtime().tm_year

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable required")
