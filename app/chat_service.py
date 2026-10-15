import json
import logging
import os
from typing import Any

from groq import Groq

from app.forecast_tool import run_forecast_via_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MarketPulse, a candid financial assistant backed by a volatility-forecast API.
You must:
- Use the run_forecast tool when the user asks about future prices, outlook, risk, or a ticker.
- The tool does not predict direction. predicted_prices is just the last close; never present it as a target.
- Explain predicted_price_low/high as an 80% range and the _95 fields as a 95% range, calibrated on
  past data. expected_move_pct is a one-standard-deviation move to each date.
- If asked whether a stock will go up or down, say the model has no directional edge.
- Never claim guaranteed returns. If asked for advice you cannot support with data, say so plainly.
- Keep answers concise unless the user asks for detail.
"""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_forecast",
            "description": "Forecast expected volatility and calibrated 80%/95% price ranges for a US ticker or index symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker e.g. AAPL, MSFT, or ^GSPC"},
                    "horizon_days": {
                        "type": "integer",
                        "description": "Trading days ahead (1-30 for pro; respect user tier on backend)",
                        "default": 7,
                    },
                },
                "required": ["symbol"],
            },
        },
    }
]


def run_chat_turn(messages: list[dict[str, Any]]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Chat is unavailable: set GROQ_API_KEY on the server."

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_CHAT_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    conv: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    for _ in range(6):
        resp = client.chat.completions.create(
            model=model,
            messages=conv,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.25,
            max_tokens=1200,
        )
        choice = resp.choices[0]
        msg = choice.message
        if getattr(msg, "tool_calls", None):
            assistant_tool_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
            conv.append(assistant_tool_msg)
            for tc in msg.tool_calls:
                if tc.function.name != "run_forecast":
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": "unknown tool"}),
                        }
                    )
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    sym = str(args.get("symbol", "")).strip()
                    hz = int(args.get("horizon_days", 7))
                    out = run_forecast_via_tool(sym, hz)
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(out.model_dump()),
                        }
                    )
                except Exception as exc:
                    logger.exception("tool run_forecast failed")
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": str(exc)}),
                        }
                    )
            continue

        return (msg.content or "").strip() or "I could not generate a response."

    return "Stopped after too many tool rounds; try a simpler question."
