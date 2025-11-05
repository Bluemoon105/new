from services.db_service import get_postgres_connection

def get_user_info(user_id:int):
    """유저 기본 정보 조회"""
    conn = get_postgres_connection()
    cur = conn.cursor()
    
     query = """
        SELECT id, name, age, gender
        FROM users
        WHERE id = %s
    """