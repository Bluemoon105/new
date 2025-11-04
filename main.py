from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sleep_api, sleepchat_api

app = FastAPI(title="Sleep Quality, Fatigue, and Optimal Sleep API")

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sleep_api.router)
app.include_router(sleepchat_api.router)