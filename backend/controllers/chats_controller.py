from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List

router = APIRouter(
    prefix="/chats",
    tags=["Chats"]
)

class ChatType(BaseModel):
    id: int
    title: str

class CreateChatResponse(BaseModel):
    id: int

# In-memory store for chats
chats_db: List[ChatType] = []
chat_id_counter = 1

@router.get("/", response_model=List[ChatType], summary="Get all chats")
async def get_chats():
    """Returns a list of all existing chats."""
    return chats_db

@router.post("/", response_model=CreateChatResponse, summary="Create a new chat")
async def create_chat():
    """Creates a new chat session and returns its ID."""
    global chat_id_counter
    new_chat = ChatType(id=chat_id_counter, title=f"New Chat {chat_id_counter}")
    chats_db.append(new_chat)
    chat_id_counter += 1
    return CreateChatResponse(id=new_chat.id)

@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a chat")
async def delete_chat(chat_id: int):
    """Deletes a chat session by ID."""
    global chats_db
    initial_length = len(chats_db)
    chats_db = [c for c in chats_db if c.id != chat_id]
    
    if len(chats_db) == initial_length:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return

@router.post("/{chat_id}/title", response_model=ChatType, summary="Update chat title")
async def update_chat_title(chat_id: int):
    """
    Updates the title of a chat organically. 
    Currently just appends a timestamp or randomizes it as a mock implementation 
    since standard API expects no body text.
    """
    import datetime
    global chats_db
    
    for chat in chats_db:
        if chat.id == chat_id:
            chat.title = f"Chat Title Updated: {datetime.datetime.now().strftime('%H:%M:%S')}"
            return chat
            
    raise HTTPException(status_code=404, detail="Chat not found")
