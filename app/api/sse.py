import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from ..core.db import SessionLocal
from ..core.deps import get_user_for_stream
from ..core.events import subscribe, unsubscribe
from ..models import Analysis, AnalysisStatus, User
from ..services.inference_runner import get_current_stage

router = APIRouter(prefix="/analyses", tags=["sse"])


@router.get("/{analysis_id}/events")
async def analysis_events(
    analysis_id: int,
    user: User = Depends(get_user_for_stream),
):
    # Open a short-lived session for the auth + initial-status snapshot, then
    # close it. Keeping it open for the whole stream would pin a DB
    # connection per active client.
    db = SessionLocal()
    try:
        analysis = db.get(Analysis, analysis_id)
        if not analysis:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        if analysis.user_id != user.id and user.role != "admin":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        initial_status = analysis.status
    finally:
        db.close()

    channel = f"analysis:{analysis_id}"
    queue = await subscribe(channel)

    # Prefer the live micro-stage (loading/inferring/rendering) over the coarse
    # row status so a late-joining client renders the right step.
    live_stage = get_current_stage(analysis_id)
    initial_stage = live_stage or initial_status.value

    async def event_gen():
        try:
            yield {"event": "status", "data": json.dumps({"stage": initial_stage})}
            if initial_status in {AnalysisStatus.completed, AnalysisStatus.failed}:
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
                    continue
                yield {"event": event.get("stage", "message"), "data": json.dumps(event)}
                if event.get("stage") in {"done", "failed"}:
                    return
        finally:
            await unsubscribe(channel, queue)

    return EventSourceResponse(event_gen())
