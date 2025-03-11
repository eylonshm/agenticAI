from __future__ import annotations
from typing import List, Dict, Any
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

website_name = os.getenv('DOCS_WEBSITE_NAME')

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

app = FastAPI(title=f"{website_name} RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    message_history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# Store conversations in memory (replace with database in production)
conversations = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print('request', request)
    conversation_id = request.conversation_id or datetime.now().isoformat()
    
    # Get or initialize conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Convert message history format if provided
    message_history = []
    if request.conversation_id and not request.message_history:
        message_history = conversations[conversation_id]
    else:
        # Convert the provided message history to the format expected by the agent
        for msg in request.message_history:
            if msg.get("role") == "user":
                message_history.append(
                    ModelRequest(parts=[UserPromptPart(content=msg.get("content", ""))])
                )
            elif msg.get("role") == "assistant":
                message_history.append(
                    ModelResponse(parts=[TextPart(content=msg.get("content", ""))])
                )
    
    # Add the new user message
    user_message = ModelRequest(parts=[UserPromptPart(content=request.message)])
    message_history.append(user_message)
    
    try:
        # Prepare dependencies
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )
        
        # Run the agent
        response_text = await run_agent(request.message, message_history, deps)
        
        # Add the response to the conversation history
        model_response = ModelResponse(parts=[TextPart(content=response_text)])
        message_history.append(model_response)
        
        # Update the conversation history
        conversations[conversation_id] = message_history
        
        print('response_text', response_text)
        print('conversations', conversations)
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_agent(user_input: str, message_history: List, deps: PydanticAIDeps) -> str:
    """
    Run the agent and return the full response text.
    """
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=message_history[:-1],  # Exclude the current user message
    ) as result:
        # Collect the full response
        full_response = ""
        async for chunk in result.stream_text(delta=True):
            full_response += chunk
            
        return full_response

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Convert the internal message format to a simpler format for the API
    formatted_messages = []
    for msg in conversations[conversation_id]:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if part.part_kind == 'user-prompt':
                    formatted_messages.append({
                        "role": "user",
                        "content": part.content
                    })
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if part.part_kind == 'text':
                    formatted_messages.append({
                        "role": "assistant",
                        "content": part.content
                    })
    
    return {"conversation_id": conversation_id, "messages": formatted_messages}

@app.get('/')
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 