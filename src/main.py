import os
from LLM.LLMClient import generate_gemini_response
from LLM.prompts.prompts import SYSTEM_PROMPT, USER_PROMPT


# ------------------------------------------------------------------------------
# Gemini Client
# ------------------------------------------------------------------------------
def gemini_response():
    response = generate_gemini_response(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
        prompt=USER_PROMPT,
        system_message=SYSTEM_PROMPT,
    )
    print(response)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    gemini_response()
