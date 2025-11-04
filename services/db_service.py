from sqlalchemy import create_engine, text
from config import settings

engine = create_engine(settings.POSTGRES_URL)

def get_user_info(user_id: int):
    """PostgreSQL에서 사용자 기본정보 조회"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT name, gender, birth_date FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        ).mappings().first()
        if not result:
            return {}
        # 생년월일 → 나이 계산
        from datetime import date
        birth = result["birth_date"]
        today = date.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return {**result, "age": age}

def get_daily_activity(user_id: int):
    """PostgreSQL에서 하루 활동 데이터 조회"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT sleep_hours, predicted_fatigue_score, recommended_sleep_range, predicted_sleep_quality
                       caffeine_mg, alcohol_consumption, physical_activity_hours
                FROM daily_activities
                WHERE user_id = :user_id
                ORDER BY created_at DESC LIMIT 1
            """),
            {"user_id": user_id}
        ).mappings().first()
        return result or {}
