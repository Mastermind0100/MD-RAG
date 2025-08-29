from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from rag import RAGbot

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
chatbot = RAGbot()

# Enable CORS for testing (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return FileResponse("templates/index.html")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    print(user_message)
    response = chatbot.chat(user_message)
    return {"reply": response}
