import time
import uuid
from typing import List, Dict, Any
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME
from .utils import hash_password, check_password

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
chats_col = db["chats"]

def create_user(username: str, password: str) -> bool:
    if users_col.find_one({"username": username}):
        return False
    users_col.insert_one({"username": username, "password": hash_password(password)})
    return True

def authenticate_user(username: str, password: str) -> bool:
    user = users_col.find_one({"username": username})
    return bool(user and check_password(password, user["password"]))

def create_chat(username: str, title: str = "New Chat") -> dict:
    chat_id = str(uuid.uuid4())
    chat = {
        "_id": chat_id,
        "username": username,
        "title": title,
        "type": "research",
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time(),
        "report_md": None,
        "papers": [],           # papers used for full report (saved after generating report)
        "ranked_papers": [],    # serialized ranked papers stored when "Run Research" finishes
        "meta": {},
        "qa_history": []        # list of {"q":..., "a":...}
    }
    chats_col.insert_one(chat)
    return chat

def get_user_chats(username: str) -> List[dict]:
    return list(chats_col.find({"username": username}).sort("updated_at", -1))

def get_chat(chat_id: str) -> dict:
    return chats_col.find_one({"_id": chat_id})

def update_chat_report_and_papers(chat_id: str, report_md: str, papers: List[Dict[str, Any]], meta: Dict[str, Any] = None) -> None:
    chats_col.update_one({"_id": chat_id}, {"$set": {
        "report_md": report_md,
        "papers": papers,
        "meta": meta or {},
        "updated_at": time.time()
    }})

def update_chat_ranked_papers(chat_id: str, ranked_papers: List[Dict[str, Any]], meta: Dict[str, Any] = None) -> None:
    """Save ranked papers (serializable dicts) to chat and update meta/updated_at."""
    existing = chats_col.find_one({"_id": chat_id}) or {}
    existing_meta = existing.get("meta", {}) if isinstance(existing.get("meta", {}), dict) else {}
    merged_meta = {**existing_meta, **(meta or {})}
    chats_col.update_one({"_id": chat_id}, {"$set": {
        "ranked_papers": ranked_papers,
        "meta": merged_meta,
        "updated_at": time.time()
    }})

def append_chat_qa(chat_id: str, q: str, a: str) -> None:
    """Append one Q/A pair to the chat's qa_history and bump updated_at."""
    chats_col.update_one({"_id": chat_id}, {"$push": {"qa_history": {"q": q, "a": a}}, "$set": {"updated_at": time.time()}})

def save_outputs_db(chat_id: str, papers: List[Any], report_md: str, out_base: str = None) -> None:
    papers_list = [p.to_row() for p in papers]
    update_chat_report_and_papers(chat_id, report_md or "", papers_list, meta={"saved_at": time.time()})
    return None
