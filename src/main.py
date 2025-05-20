import os
from LLM.LLMClient import generate_gemini_response
from google import genai
from LLM.prompts.prompts import SYSTEM_PROMPT, USER_PROMPT
from models.BaseModel import Recipe


# ------------------------------------------------------------------------------
# Gemini Client normal
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
# Structured Output
# ------------------------------------------------------------------------------

def structured_gemini_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="List a few popular cookie recipes, and include the amounts of ingredients.",
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Recipe],
        },
    )

    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    my_recipes: list[Recipe] = response.parsed
    print(my_recipes)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # gemini_response()

    structured_gemini_response()    
