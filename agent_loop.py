"""
agent_loop.py
-------------
Hand-rolled tool-call loop with short-term memory and a step limit.
No agent framework — plain Python + google-genai SDK.

Usage:
    cp .env.example .env        # add your GOOGLE_API_KEY
    python agent_loop.py
"""

import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ── Load API key from .env (NEVER hardcode secrets!) ─────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found.\n"
        "  1. cp .env.example .env\n"
        "  2. Paste your Gemini key into .env\n"
        "  3. Run again."
    )

client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"

# ── Load order data ───────────────────────────────────────────────────────────
ORDERS_FILE = Path(__file__).parent / "orders.json"
with ORDERS_FILE.open() as f:
    ORDERS: dict = json.load(f)


# ── Tool implementations (plain Python functions) ─────────────────────────────

def lookup_order(order_id: str) -> dict:
    """Return order details for the given order ID."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"Order '{order_id}' not found."}
    return {
        "order_id":        order_id.upper(),
        "item":            order["item"],
        "price":           order["price"],
        "purchased":       order["purchased"],
        "warranty_months": order["warranty_months"],
    }


def calculate(expression: str) -> dict:
    """Safely evaluate a simple arithmetic expression and return the result."""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return {"error": "Unsafe expression — only basic arithmetic is allowed."}
    try:
        result = eval(expression, {"__builtins__": {}})   # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


# Registry: tool name → Python function
TOOL_REGISTRY = {
    "lookup_order": lookup_order,
    "calculate":    calculate,
}

# ── Tool declarations for the Gemini API ──────────────────────────────────────
TOOL_CONFIG = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="lookup_order",
            description=(
                "Look up an order by its ID (e.g. A1001) and return the item name, "
                "price, purchase date, and warranty length."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "order_id": types.Schema(
                        type=types.Type.STRING,
                        description="The order ID, e.g. A1001",
                    )
                },
                required=["order_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluate a simple arithmetic expression such as '1200 * 3' "
                "and return the numeric result."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="A safe arithmetic expression using numbers and +-*/().",
                    )
                },
                required=["expression"],
            ),
        ),
    ]
)


# ── The hand-rolled agent loop ────────────────────────────────────────────────

def run_agent(user_message: str, messages: list, max_steps: int = 5) -> str:
    """
    Append user_message to `messages`, run the model→tool→model loop, and
    return the final text answer.  `messages` is mutated in place — this is
    the short-term memory that persists across turns.

    Parameters
    ----------
    user_message : the new user turn
    messages     : running conversation history (modified in place)
    max_steps    : hard cap on model↔tool iterations before giving up
    """
    # 1. Add the new user turn to memory
    messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    print(f"\n{'═'*62}")
    print(f"  USER  ▶  {user_message}")
    print(f"{'═'*62}")

    for step in range(1, max_steps + 1):
        print(f"\n  [step {step}/{max_steps}]  Sending {len(messages)} message(s) to model …")

        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[TOOL_CONFIG],
                temperature=0,
            ),
        )

        candidate  = response.candidates[0]
        reply_content = candidate.content          # types.Content
        parts         = reply_content.parts or []

        # 2. Append the model's raw reply to memory (preserves tool calls too)
        messages.append(reply_content)

        # ── Are there tool calls in this reply? ───────────────────────────────
        fn_calls = [p for p in parts if p.function_call is not None]

        if fn_calls:
            tool_response_parts = []
            for part in fn_calls:
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args)
                print(f"  [step {step}]  Tool call  → {fn_name}({fn_args})")

                if fn_name not in TOOL_REGISTRY:
                    result = {"error": f"Unknown tool: {fn_name}"}
                else:
                    result = TOOL_REGISTRY[fn_name](**fn_args)

                print(f"  [step {step}]  Tool result← {result}")

                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response={"result": result},
                        )
                    )
                )

            # 3. Append all tool results as a single "tool" turn and loop back
            messages.append(
                types.Content(role="tool", parts=tool_response_parts)
            )
            continue  # ← go back to model with the tool results

        # ── No tool calls → this is the final text answer ─────────────────────
        text_parts = [p.text for p in parts if p.text]
        final_text = " ".join(text_parts).strip()
        print(f"\n  MODEL ◀  {final_text}")
        return final_text

    # ── Step limit hit ────────────────────────────────────────────────────────
    msg = f"[Agent stopped: couldn't finish within {max_steps} steps]"
    print(f"\n  ⚠  {msg}")
    return msg


# ── Two-turn memory demo ──────────────────────────────────────────────────────

def main():
    # One shared list = the agent's short-term memory for the whole session
    messages: list = []

    print("\n" + "━"*62)
    print("  DEMO: Two-turn conversation proving short-term memory")
    print("━"*62)

    # Turn 1 — look up order price
    answer1 = run_agent(
        "What did order A1001 cost?",
        messages,
    )

    # Turn 2 — depends entirely on the price learned in Turn 1
    #           Only works because `messages` still holds that context
    answer2 = run_agent(
        "And what about three of them?",
        messages,
    )

    print("\n" + "━"*62)
    print("  FINAL SUMMARY")
    print("━"*62)
    print(f"  Turn 1 : {answer1}")
    print(f"  Turn 2 : {answer2}")
    print(f"  Memory : {len(messages)} content blocks in history")
    print("━"*62 + "\n")


if __name__ == "__main__":
    main()
