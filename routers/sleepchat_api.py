from fastapi import APIRouter
from services.llm_service import generate_sleep_feedback
from services.mongo_service import save_chat, get_user_chats
from models.sleepSchema import ChatRequest, SleepChatRequest

router = APIRouter(prefix="/sleepchat", tags=["Chat"])

@router.post("/message")
def chat_general(req: ChatRequest):
    """일상 대화"""
    response = f"'{req.message}'에 대한 일상적인 응답입니다 😊"
    save_chat(req.user_id, req.message, response, chat_type="general")
    return {"response": response}


@router.post("/sleep-feedback")
def chat_sleep_feedback(req: SleepChatRequest):
    """수면 데이터 기반 피드백"""
    # req 전체를 통째로 넘겨야 함
    response = generate_sleep_feedback(req)
    save_chat(
        req.user_id,
        f"수면 질:{req.sleep_quality}, 피로도:{req.fatigue_score}, 추천:{req.recommended_range}",
        response,
        chat_type="sleep"
    )
    return {"response": response}


@router.get("/history/{user_id}")
def get_chat_history(user_id: int):
    """특정 유저의 최근 대화 기록"""
    chats = get_user_chats(user_id)
    return {"user_id": user_id, "history": chats}
