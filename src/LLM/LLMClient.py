import google.generativeai as genai

def generate_gemini_response(
    api_key: str,
    model_name: str,
    prompt: str,
    system_message: str = None,
    temperature: float = None,
    max_output_tokens: int = None,
) -> str:
    """
    Sends a prompt to the Gemini LLM and gets a simple text response.
    This function configures the API key, creates the model, and generates content.

    Args:
        prompt (str): The user's prompt.
        api_key (str): Your Google API key. Must be provided.
        model_name (str): The name of the Google Gemini model to use.
        system_message (str, optional): System instruction for the model.
        temperature (float, optional): Controls randomness (0.0-1.0).
        max_output_tokens (int, optional): Maximum number of tokens to generate.

    Returns:
        str: The generated text content from the LLM.
             Returns an empty string if the model generates no text (e.g. due to safety filters).

    Raises:
        ValueError: If api_key is not provided.
        google.api_core.exceptions.GoogleAPIError: For API-related errors during generation.
        Other exceptions from the google.generativeai library may also propagate.
    """
    genai.configure(api_key=api_key)
    if not api_key:
        raise ValueError(
            "A Google API key must be provided to 'generate_gemini_response'."
        )

    model_instance = genai.GenerativeModel(
        model_name=model_name, system_instruction=system_message
    )

    generation_config_params = {}
    if temperature is not None:
        generation_config_params["temperature"] = temperature
    if max_output_tokens is not None:
        generation_config_params["max_output_tokens"] = max_output_tokens

    current_generation_config = (
        genai.types.GenerationConfig(**generation_config_params)
        if generation_config_params
        else None
    )

    response = model_instance.generate_content(
        contents=prompt,
        generation_config=current_generation_config
    )

    return response.text
