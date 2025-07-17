from typing import Dict, Any, Tuple
import json
import os
import yaml
import ast
import re
from openapi_spec_validator import validate_spec
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# --- Validation Logic ---


def validate_openapi_spec(spec_content: str) -> Tuple[bool, list]:
    """
    Validate the OpenAPI spec (YAML or JSON). Returns (is_valid, errors).
    """
    try:
        if spec_content.strip().startswith('{'):
            spec = json.loads(spec_content)
        else:
            spec = yaml.safe_load(spec_content)
    except Exception as e:
        return False, [f"Parsing error: {str(e)}"]
    try:
        validate_spec(spec)
        return True, []
    except Exception as e:
        return False, [str(e)]

# --- LangGraph Correction Flow ---


def correct_openapi_with_llm(original_content: str, errors: list, llm=None) -> Dict[str, Any]:
    """
    Use LangGraph and LLM to correct the OpenAPI spec based on errors.
    Returns dict with corrected YAML, JSON, and explanations.
    """
    if llm is None:
        llm = get_llm()
    prompt = f"""You are an expert in OpenAPI/Swagger specifications.

    The following OpenAPI spec contains validation errors.

    Your tasks:
    1. Correct the OpenAPI spec.
    2. Provide the corrected spec in both YAML and JSON formats.
    3. Explain each change made.
    4. Return **only** a Python dictionary with the following structure and keys:

    {{
        'yaml': '<corrected YAML string>',
        'json': '<corrected JSON string>',
        'explanations': ['<explanation 1>', '<explanation 2>', ...],
        'raw_response': '<full original raw output including YAML, JSON, and explanations>'
    }}

    IMPORTANT:
    - Do NOT include markdown code blocks like ```yaml or ```python
    - Do NOT include explanations or narrative text before or after the dictionary
    - Return ONLY the dictionary object as plain text

    Input:
    ---
    Original Spec:
    {original_content}

    Validation Errors:
    {errors}
    """
    print('-----------------------------------')
    print(prompt)
    print('-----------------------------------')
    response = llm.invoke(prompt)
    yaml_part, json_part, explanations = '', '', []
    print("----- RAW LLM RESPONSE START -----")
    print(repr(response.content))  # use repr to reveal hidden characters
    print("----- RAW LLM RESPONSE END -----")
    print('----------------OUTPUT-------------------')
    # if 'YAML:' in response and 'JSON:' in response and 'EXPLANATIONS:' in response:
    #     try:
    #         yaml_part = response.split('YAML:')[1].split('---')[0].strip()
    #         json_part = response.split('JSON:')[1].split('---')[0].strip()
    #         explanations = [line.strip(
    #             '- ').strip() for line in response.split('EXPLANATIONS:')[1].split('\n') if line.strip()]
    #     except Exception:
    #         pass
    res = extract_llm_response_fields(response.content)
    return res


def clean_llm_response_block(content):
    # Step 1: Extract .content if LangChain object
    if hasattr(content, "content"):
        content = content.content

    # Step 2: Strip Markdown code block if present
    content = content.strip()

    # Match triple-backtick Python block
    if content.startswith("```python") and content.endswith("```"):
        content = content[len("```python"):].strip()
        content = content[:-3].strip() if content.endswith("```") else content

    return content


def extract_llm_response_fields(content):
    """
    Clean LLM output and extract structured fields.
    """
    try:
        cleaned = clean_llm_response_block(content)
        response_dict = ast.literal_eval(cleaned)

        return {
            'yaml': response_dict.get('yaml', '').strip(),
            'json': response_dict.get('json', '').strip(),
            'explanations': response_dict.get('explanations', []),
            'raw_response': response_dict.get('raw_response', '').strip()
        }

    except Exception as e:
        print("❌ Failed to parse cleaned LLM response:", e)
        return {
            'yaml': '',
            'json': '',
            'explanations': [],
            'raw_response': content if isinstance(content, str) else str(content)
        }


def run_correction_pipeline(spec_content: str, llm=None) -> Dict[str, Any]:
    is_valid, errors = validate_openapi_spec(spec_content)
    if is_valid:
        return {
            'valid': True,
            'errors': [],
            'corrected': None,
            'explanations': [],
        }
    correction = correct_openapi_with_llm(spec_content, errors, llm=llm)
    return {
        'valid': False,
        'errors': errors,
        'corrected': correction,
        'explanations': correction.get('explanations', [])
    }


__all__ = ["get_llm", "validate_openapi_spec", "run_correction_pipeline"]
