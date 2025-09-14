from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from service import get_vector_store
from prompt import system_prompt
from chatbot.chatbot_with_routing import workflow, all_tools

class UserQuestion(BaseModel):
    question: str

app = FastAPI(
    title="Chatbot API",
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent(  
    "gemini-2.5-flash",
    system_prompt=system_prompt,
)

message_history: list[ModelMessage] = []

@agent.tool
async def retrieve(ctx: RunContext, question: str) -> str:  
    """This is agent that can answer to user task related knowledge base of documents.
    
    Args:
        question: The question to be answered.
    """
    handler1 = workflow.run(
        query=question,
        tools=all_tools,
    )
    
    async for event in handler1.stream_events():
        if hasattr(event, "msg"):
            print(event.msg)
    
    result1 = await handler1
    
    return result1


@app.post("/")
async def chatbot_response( 
    request: UserQuestion,
    vector_store: None = Depends(get_vector_store),
):
    user_question = request.question.lower().strip()
    
    result = await agent.run(user_question, message_history=message_history)
    model_messages = result.new_messages()
    message_history.extend(model_messages)

    return {"response": result.output}


@app.options("/")
async def check( 
    request: Request
):
    return True


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
