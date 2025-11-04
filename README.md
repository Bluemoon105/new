# new
```
new
├─ config.py
├─ main.py
├─ models
│  ├─ best_sleep_quality_rf_bundle.pkl - 예측 ML 모델
│  ├─ daily_activity_model.py - 하루 활동량 모델
│  ├─ sleepSchema.py - 스키마
│  └─ user_model.py - 유저 모델
├─ README.md
├─ routers
│  ├─ sleepchat_api.py - 챗봇 관련 api
│  └─ sleep_api.py - 예측 관련 api
└─ services
   ├─ db_service.py - postrgreSQL 연결
   ├─ llm_service.py - llm 연결
   ├─ mongo_service.py - mongodb연결
   ├─ sleep_service.py - 예측에 필요한 함수
   └─ user_service.py - 유저 정보 불러오기

```