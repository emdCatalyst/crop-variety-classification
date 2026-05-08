from datetime import datetime

from pydantic import BaseModel


class MessageOut(BaseModel):
    id: int
    sender_id: int
    sender_name: str
    recipient_id: int
    body: str | None
    has_attachment: bool
    attachment_name: str | None
    attachment_mime: str | None
    read_at: datetime | None
    created_at: datetime
    archived: bool


class ThreadOut(BaseModel):
    thread_key: str
    other_user_id: int
    other_user_name: str
    other_user_role: str
    last_body: str | None
    last_has_attachment: bool
    last_at: datetime
    unread_count: int
    archived: bool
