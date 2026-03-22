"""
FloodBot — Groq-powered AI Chatbot for SovereignFlood India
Uses LLaMA 3.3 70B via Groq's free API (14,400 req/day, no credit card needed)

Setup:
  1. Sign up at https://console.groq.com  (free, just email)
  2. Create an API key
  3. Add to your .env file:  GROQ_API_KEY=your_key_here
  4. pip install groq python-dotenv
"""

import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# ── Client (reads GROQ_API_KEY from env) ──────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Models ────────────────────────────────────────────────────────────────────
# Use VERSATILE for most queries — it's smarter and still free (14,400 req/day)
# Use INSTANT only if you need sub-second responses (lighter model)
MODEL_VERSATILE = "llama-3.3-70b-versatile"   # best quality, free
MODEL_INSTANT   = "llama-3.1-8b-instant"      # fastest, lighter

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_BASE = """You are FloodBot — the AI assistant for SovereignFlood India, 
a ward-level flood intelligence platform used by Indian municipal officers.

YOUR ROLE:
- Help ward officers and municipal staff make flood preparedness decisions
- Explain flood risk scores, drain health, and readiness metrics clearly
- Suggest specific, costed interventions (drain desilting ₹8.5L, pump deployment ₹12L, etc.)
- Answer questions about monsoon preparedness, drainage systems, and emergency routing
- Compare flood readiness across Indian cities when asked

LANGUAGE RULE:
- If the user writes in Hindi (Devanagari script or Hinglish), respond FULLY in Hindi
- If the user writes in English, respond in English
- Never mix scripts mid-sentence

TONE & FORMAT:
- Be concise — ward officers are busy people. Max 4-5 bullet points or 3 short sentences
- Be specific — always reference actual numbers from the city data provided
- When suggesting actions, include rough cost estimates in INR
- Use ✓ for completed items, ⚠ for warnings, 🚨 for critical alerts

BOUNDARIES:
- Only answer flood management, drainage, disaster preparedness, and municipal planning questions
- For anything unrelated, say: "I can only help with flood and disaster management queries for Indian cities."
- Never make up statistics. If data isn't in your context, say "I don't have that data for this city."
"""


def build_city_context(city_data: Optional[dict]) -> str:
    """Convert live city data into a context block injected into every message."""
    if not city_data:
        return "\n[No city selected yet — user needs to select a city first]\n"

    high_risk = city_data.get("high_risk_wards", [])
    high_risk_str = ", ".join(high_risk[:5]) if high_risk else "None identified"
    
    moderate = city_data.get("moderate_wards", [])
    moderate_str = ", ".join(moderate[:4]) if moderate else "None"

    return f"""
=== LIVE CITY DATA (use this for specific answers) ===
City              : {city_data.get("city", "Unknown")}, India
Current Rainfall  : {city_data.get("rainfall_mm", 0):.1f} mm/hr
7-Day Max Forecast: {city_data.get("max_forecast_mm", 0):.1f} mm
High Risk Wards   : {high_risk_str}
Moderate Wards    : {moderate_str}
Total Wards       : {city_data.get("total_wards", 0)}
Avg City Readiness: {city_data.get("avg_readiness", 0):.0f}/100
High Risk Count   : {city_data.get("high_risk_count", 0)}
Days to Monsoon   : {city_data.get("days_to_monsoon", "N/A")}
Drains Monitored  : {city_data.get("total_drains", 0)}
Avg Drain Health  : {city_data.get("avg_drain_health", 0):.0f}/100
=== END LIVE DATA ===
"""


def get_suggested_questions(city_name: str) -> List[str]:
    """Contextual quick-reply chips shown in the chat UI."""
    return [
        f"Which wards in {city_name} are most at risk this monsoon?",
        f"What is the top intervention needed in {city_name} right now?",
        "How do I read the Ward Readiness Score?",
        f"What should {city_name} do with ₹5 crore flood budget?",
        "What actions should a ward officer take 48 hours before heavy rain?",
        f"Is {city_name} more or less prepared than last year?",
    ]


def chat(
    history: List[Dict],
    user_message: str,
    city_data: Optional[dict] = None,
    fast_mode: bool = False,
) -> str:
    """
    Send a message to Groq and return the AI reply.

    Args:
        history      : Full conversation so far as list of {"role": ..., "content": ...}
        user_message : The new message from the user
        city_data    : Live city context dict (from frontend state)
        fast_mode    : True = use 8B instant model (faster), False = use 70B (smarter)

    Returns:
        AI reply as a plain string
    """
    model = MODEL_INSTANT if fast_mode else MODEL_VERSATILE
    system_prompt = SYSTEM_BASE + build_city_context(city_data)

    # Build message list: system → history → new user message
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=450,
            temperature=0.65,       # slightly creative but mostly factual
            top_p=0.9,
            stream=False,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "429" in err:
            return "⚠ Rate limit reached. Please wait 10 seconds and try again."
        if "401" in err:
            return "⚠ API key error. Please check your GROQ_API_KEY in the .env file."
        if "model_not_found" in err.lower():
            return "⚠ Model unavailable. Please check your Groq dashboard."
        return f"⚠ FloodBot is temporarily unavailable: {err[:120]}"
