from typing import List, Dict, Optional
from pydantic import BaseModel
from typing import Any

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None