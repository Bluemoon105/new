import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    """환경변수를 불러와 전역적으로 관리하는 클래스"""

    APP_NAME = os.getenv("APP_NAME", "SleepTalk AI")
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_PORT = int(os.getenv("APP_PORT", 8000))

    POSTGRES_URL = os.getenv("POSTGRES_URL")
    MONGO_URL = os.getenv("MONGO_URL")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # KANANA_MODEL = os.getenv("KANANA_MODEL")
    SECRET_KEY = os.getenv("SECRET_KEY")

    #Hugging Face Private Model 인증 토큰
    # HF_TOKEN = os.getenv("HF_TOKEN")

settings = Settings()
