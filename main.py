import os
import re
import json
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
SHEET_WEBHOOK_URL = os.getenv("SHEET_WEBHOOK_URL", "")

# Swap this block with ANY client's website copy in 60 seconds before pitching them!
BUSINESS_KNOWLEDGE = """
Business Name: NovaScale Digital Studio
Services & Pricing:
1. High-Converting Landing Page (48-hour delivery): ₹8,000 ($99)
2. 24/7 AI Customer Support & Lead Bot Setup: ₹6,000 ($75) + ₹4,500/mo ($55/mo) care plan
3. Custom Python Workflow & Scraper Automation: Starts at ₹5,000 ($60)
Process: 50% upfront to start, 50% on live delivery. 2 free revision rounds included.
Contact / Booking: Founders can book a 15-minute discovery call or request a custom quote right in this chat.
"""

SYSTEM_INSTRUCTION = f"""
You are the 24/7 AI Concierge for the business described below.
Rules:
1. Answer ONLY using the provided Business Knowledge. If asked something outside it, politely say you don't have that exact detail and offer to connect them with the founder.
2. Keep every reply concise (2 to 3 short sentences max), friendly, and focused on helping the visitor.
3. Whenever a visitor shows buying intent, asks for a custom quote, or wants to book a call, ask for their Name and Email or WhatsApp number.
4. CRITICAL: The moment the user provides their Name and Email/WhatsApp, append this exact hidden tag at the very end of your reply (fill in their details):
[LEAD_JSON]{{"name": "...", "contact": "...", "question": "..."}}[/LEAD_JSON]

Business Knowledge:
{BUSINESS_KNOWLEDGE}
"""

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

@app.get("/")
async def serve_demo():
    return FileResponse("public/index.html")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    contents = []
    for turn in req.history[-6:]:
        contents.append(
            types.Content(
                role=turn["role"],
                parts=[types.Part.from_text(text=turn["text"])]
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=req.message)]
        )
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        ),
    )

    reply_text = response.text or "Sorry, I couldn't process that. Please try again."

    # Check if Gemini captured a lead and forward it to Google Sheets
    lead_match = re.search(r"\[LEAD_JSON\](.*?)\[/LEAD_JSON\]", reply_text, re.DOTALL)
    if lead_match:
        raw_json = lead_match.group(1).strip()
        reply_text = re.sub(r"\[LEAD_JSON\].*?\[/LEAD_JSON\]", "", reply_text, flags=re.DOTALL).strip()
        if SHEET_WEBHOOK_URL:
            try:
                lead_data = json.loads(raw_json)
                async with httpx.AsyncClient(timeout=8.0) as http_client:
                    await http_client.post(SHEET_WEBHOOK_URL, json=lead_data)
            except Exception as e:
                print("Webhook error:", e)

    return {"reply": reply_text}