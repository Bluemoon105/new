# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import settings
from services.user_service import get_user_info, get_daily_activity, get_weekly_activity
from services.mongo_service import save_chat, get_user_chats
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")



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
model = genai.GenerativeModel("gemini-2.5-flash")

def _safe_avg(items, key, ndigits=None):
    vals = [x.get(key) for x in items if x.get(key) is not None]
    if not vals:
        return "N/A"
    avg = sum(vals) / len(vals)
    return round(avg, ndigits) if ndigits is not None else avg

# 유틸: 빈 값 제거
def _fmt(label: str, value, unit: str = ""):
    if not value or value in ["N/A", "None", "비공개"]:
        return ""
    return f"- {label}: {value}{unit}"

#일반 수면 대화용
@traceable
def generate_general_chat(req):
    """일상 수면 관련 대화 (피로도 데이터 없이 일반 질문 대응)"""
    prompt = f"""
    너는 수면 코치지만, 친구처럼 편하게 대화해주는 챗봇이야.
    사용자가 수면, 스트레스, 피로, 루틴, 휴식 등에 대해 물어보면
    따뜻하고 공감가는 말투로 대답해줘.

    단, 너무 과학적이거나 의학적인 설명은 피하고,
    짧고 부드러운 문장으로 이야기하듯 답해줘 😊

    사용자 질문: {req.message}
    """

    result = model.generate_content(prompt)
    response = result.text.strip()

    save_chat(req.user_id, req.message, response, chat_type="general")
    return response

# 일간 리포트 (하루 데이터 기반)
@traceable
def generate_daily_report(user_id:int):
    user = get_user_info(user_id) or {}
    activity = get_daily_activity(user_id) or {}
    
    if not activity:
        return "오늘의 수면 데이터가 아직 없습니다"
    
    prompt = f"""
    너는 전문적인 수면 코치야 
    아래의 사용자 하루 수면 데이터를 분석해서 '오늘의 수면 리포트'를 작성해줘.
    
    [사용자]
    이름: {user.get('name', '비공개')}
    나이: {user.get('age', '비공개')}세
    
     [오늘의 수면 데이터]
    수면시간: {activity.get('sleep_hours', 'N/A')}시간
    피로도 점수: {activity.get('predicted_fatigue_score', 'N/A')}
    수면 질 예측: {activity.get('predicted_sleep_quality', 'N/A')}
    카페인 섭취: {activity.get('caffeine_mg', 'N/A')}mg
    알코올 섭취: {activity.get('alcohol_consumption', 'N/A')}회
    활동량: {activity.get('physical_activity_hours', 'N/A')}시간
    추천 수면시간: {activity.get('recommended_sleep_range', 'N/A')}
    
    ---
    출력 형식:
    1.수면 상태 요약
    2. 피로도 및 원인 분석
    3. 개선 팁 2~3개
    4. 짧은 응원 메시지
    """
    
    result = model.generate_content(prompt)
    response = result.text.strip()
    
    save_chat(user_id, "일간 리포트 요청", response, chat_type="report")
    return response

#주간 리포트
@traceable
def generate_weekly_report(user_id:int):
    user = get_user_info(user_id) or {}
    week_data = get_weekly_activity(user_id) or {}
    
    if not week_data:
        return "최근 7일간의 수면 데이터가 부족합니다. 최고1개 이상의 기록이 필오해요"
    
    # 평균 계산
    avg_sleep = _safe_avg(week_data, "sleep_hours", 2)
    avg_fatigue =  _safe_avg(week_data, "predicted_fatifue_score",1)
    avg_quality = _safe_avg(week_data, "predicted_sleep_quality", 2)
    avg_activity = _safe_avg(week_data, "physical-activity_hours", 1)
    
    def _is_weekend(row):
        d = row.get("date")
        try:
            return d.weekday() >=5
        except Exception:
            return None
        
    weekdays = [r for r in week_data if _is_weekend(r) is False]
    weekends = [r for r in week_data if _is_weekend(r) is True]
    
    weekday_sleep = _safe_avg(weekdays, "sleep_hours", 2) if weekdays else "N/A"
    weekend_sleep = _safe_avg(weekends, "sleep_hours", 2) if weekends else "N/A"
    
    prompt = f"""
    너는 전문 수면 분석 코치야.
    아래의 최근 7일 데이터를 바탕으로 '주간 수면 리포트'를 만들어줘.
    
    [사용자]
    이름: {user.get('name', '비공개')}
    나이: {user.get('age', '비공개')}세
    
    [집계 범위]
    기록 수 {len(week_data)}건 (최대 7건)

    [7일 평균 데이터]
    평균 수면시간: {avg_sleep}시간
    평균 피로도 점수: {avg_fatigue}
    평균 수면 질 점수: {avg_quality}
    평균 활동량: {avg_activity}시간
    
    [패턴 힌트]
    - 평일 평균 수면시간: {weekday_sleep}시간
    - 주말 평균 수면시간: {weekend_sleep}시간
    
    ---
    출력 형식:
    1. 주간 수면 요약 (좋았던 점, 아쉬운 점)
    2. 패턴 분석 (평일 vs 주말)
    3. 다음 주 개선 팁 2가지
    4. 끝에 짧은 응원 한마디
    """
    
    result = model.generate_content(prompt)
    response = result.text.strip()
    
    save_chat(user_id, "주간 리포트 요청", response, chat_type="report")
    return response

# KANANA
# def generate_sleep_feedback(req):
#     user_id = req.user_id
#     user = get_user_info(user_id) or {}
#     activity = get_daily_activity(user_id) or {}
#     history = get_user_chats(user_id, limit=5) or []

#     history_text = "\n".join(
#         [f"사용자: {h['user_message']} / 코치: {h['bot_response']}" for h in history]
#     ) or "최근 대화 없음"

#     user_info = "\n".join(filter(None, [
#         _fmt("이름", user.get("name")),
#         _fmt("나이", user.get("age"), "세"),
#         _fmt("성별", user.get("gender")),
#     ]))

#     activity_info = "\n".join(filter(None, [
#         _fmt("수면시간", activity.get("sleep_hours"), "시간"),
#         _fmt("피로도 점수", activity.get("fatigue_score")),
#         _fmt("카페인 섭취량", activity.get("caffeine_mg"), "mg"),
#         _fmt("알코올 섭취량", activity.get("alcohol_consumption"), "회"),
#         _fmt("활동량", activity.get("physical_activity_hours"), "시간"),
#         _fmt("추천 수면 시간", activity.get("recommended_range")),
#     ]))

#     #핵심: '예시 출력' '규칙' 등 제거하고, 답변 시작 지점 명확히 지정
#     prompt = f"""
#         너는 따뜻하고 친절한 수면 코치야.
#         데이터를 기반으로 간결하고 따뜻한 피드백을 줘.

#         [사용자 정보]
#         {user_info or '- 정보 없음'}

#         [최근 하루 데이터]
#         {activity_info or '- 데이터 없음'}

#         [최근 대화]
#         {history_text}

#         [사용자 질문]
#         {req.message}

#         ---

#         다음 내용을 반드시 포함해서 자연스럽게 말해줘:
#         1. 상태 요약 (현재 수면 상태 한 줄)
#         2. 오늘의 컨디션 코멘트
#         3. 실천 가능한 수면 개선 팁 2~3개 (불릿 형식)
#         4. 마지막에 따뜻한 응원의 말

#         응답은 아래 형식을 참고하되, 예시는 출력하지 마.
#         ### 답변 시작:
#         """

#     result = generator(
#         prompt,
#         max_new_tokens=250,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         eos_token_id=TOKENIZER.eos_token_id,
#     )

#     response = result[0]["generated_text"].strip()

#     save_chat(user_id, req.message, response, chat_type="sleep")
#     return response
    