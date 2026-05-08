import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..core.db import get_db
from ..core.deps import get_current_user
from ..core.events import subscribe, unsubscribe
from ..models import Analysis, AnalysisStatus, User

router = APIRouter(prefix="/analyses", tags=["sse"])


@router.get("/{analysis_id}/events")
async def analysis_events(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if analysis.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    channel = f"analysis:{analysis_id}"
    queue = await subscribe(channel)

    initial_status = analysis.status

    async def event_gen():
        try:
            yield {"event": "status", "data": json.dumps({"stage": initial_status.value})}
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
