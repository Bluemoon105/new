from pymongo import MongoClient
from datetime import datetime
from config import settings

client = MongoClient(settings.MONGO_URL)
db = client[settings.MONGO_DB_NAME]
chat_collection = db["chat_history"]

def save_chat(user_id: int, user_message: str, bot_response: str, chat_type: str = "general"):
    """대화 내용을 MongoDB에 저장"""
    chat_doc = {
        "user_id": user_id,
        "chat_type": chat_type,   # 'general' 또는 'sleep'
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow(),
    }
    chat_collection.insert_one(chat_doc)

def get_user_chats(user_id: int, limit: int = 5):
    """특정 유저의 최근 대화 가져오기"""
    return list(
        chat_collection.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(limit)
    )
