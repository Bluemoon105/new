# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import settings
from services.user_service import get_user_info, get_daily_activity
from services.mongo_service import save_chat, get_user_chats
import google.generativeai as genai

# MODEL_NAME = settings.KANANA_MODEL
# TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, token=settings.HF_TOKEN)
# MODEL = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype="auto",
#     device_map="auto",
#     token=settings.HF_TOKEN,
# )
# generator = pipeline("text-generation", model=MODEL, tokenizer=TOKENIZER, device_map="auto")

genai.configure(api_key=settings.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 유틸: 빈 값 제거
def _fmt(label: str, value, unit: str = ""):
    if not value or value in ["N/A", "None", "비공개"]:
        return ""
    return f"- {label}: {value}{unit}"


def generate_sleep_feedback(req):
    user_id = req.user_id
    user = get_user_info(user_id) or {}
    activity = get_daily_activity(user_id) or {}
    history = get_user_chats(user_id, limit=5) or []

    history_text = "\n".join(
        [f"사용자: {h['user_message']} / 코치: {h['bot_response']}" for h in history]
    ) or "최근 대화 없음"

    user_info = "\n".join(filter(None, [
        _fmt("이름", user.get("name")),
        _fmt("나이", user.get("age"), "세"),
        _fmt("성별", user.get("gender")),
    ]))

    activity_info = "\n".join(filter(None, [
        _fmt("수면시간", activity.get("sleep_hours"), "시간"),
        _fmt("피로도 점수", activity.get("fatigue_score")),
        _fmt("카페인 섭취량", activity.get("caffeine_mg"), "mg"),
        _fmt("알코올 섭취량", activity.get("alcohol_consumption"), "회"),
        _fmt("활동량", activity.get("physical_activity_hours"), "시간"),
        _fmt("추천 수면 시간", activity.get("recommended_range")),
    ]))

    #핵심: '예시 출력' '규칙' 등 제거하고, 답변 시작 지점 명확히 지정
    prompt = f"""
        너는 따뜻하고 친절한 수면 코치야.
        데이터를 기반으로 간결하고 따뜻한 피드백을 줘.

        [사용자 정보]
        {user_info or '- 정보 없음'}

        [최근 하루 데이터]
        {activity_info or '- 데이터 없음'}

        [최근 대화]
        {history_text}

        [사용자 질문]
        {req.message}

        ---

        다음 내용을 반드시 포함해서 자연스럽게 말해줘:
        1. 상태 요약 (현재 수면 상태 한 줄)
        2. 오늘의 컨디션 코멘트
        3. 실천 가능한 수면 개선 팁 2~3개 (불릿 형식)
        4. 마지막에 따뜻한 응원의 말

        응답은 아래 형식을 참고하되, 예시는 출력하지 마.
        ### 답변 시작:
        """

    result = generator(
        prompt,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
        eos_token_id=TOKENIZER.eos_token_id,
    )

    response = result[0]["generated_text"].strip()

    save_chat(user_id, req.message, response, chat_type="sleep")
    return response
