from __future__ import annotations

from pathlib import Path

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.config import settings
from app.database.mongo import ensure_indexes, get_database
from app.realtime import ws_manager
from app.routes.admin import router as admin_router
from app.routes.attendance import router as attendance_router
from app.routes.auth import router as auth_router
from app.routes.teacher import router as teacher_router
from app.routes.student import router as student_router
from app.services.attendance_service import send_absent_notifications
from app.utils.time import now_ist
from app.utils.time import iso_ist


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    app.state.absent_notices_task = asyncio.create_task(_absent_notices_worker())
    try:
        yield
    finally:
        task = getattr(app.state, "absent_notices_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="Smart Attendance System Using Face Recognition API", lifespan=lifespan)
app.state.ws_manager = ws_manager
app.state.absent_notices_task = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_file = FRONTEND_DIR / "favicon.svg"
    if icon_file.exists():
        return FileResponse(icon_file, media_type="image/svg+xml")
    return Response(status_code=204)


async def _absent_notices_worker() -> None:
    while True:
        try:
            if not settings.absent_notifications_enabled:
                await asyncio.sleep(60)
                continue

            current = now_ist()
            cutoff_passed = (current.hour, current.minute) >= (settings.absent_notice_hour, settings.absent_notice_minute)
            if cutoff_passed:
                db = await get_database()
                departments = await db.users.distinct("department", {"role": "student", "department": {"$ne": None}})
                for department in departments:
                    result = await send_absent_notifications(db, department)
                    if result.get("status") == "failed":
                        print(f"[ABSENT:FAILED] {department} | {result.get('message')}")
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(60)


@app.get("/")
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return RedirectResponse(url="/static/index.html")
    return {"message": "Smart Attendance System Using Face Recognition API is operational."}


@app.get("/health")
async def health():
    return {"status": "ok", "time": iso_ist()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


app.include_router(auth_router)
app.include_router(attendance_router)
app.include_router(teacher_router)
app.include_router(student_router)
app.include_router(admin_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

