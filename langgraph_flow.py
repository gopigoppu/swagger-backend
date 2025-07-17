from typing import Dict, Any, Tuple
import json
import os
import yaml
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
    prompt = f"""
You are an expert in OpenAPI/Swagger specifications. The following OpenAPI spec has validation errors. Please correct the errors and explain each change you make.\n\nOriginal Spec:\n{original_content}\n\nErrors:\n{errors}\n\nReturn the corrected spec in both YAML and JSON, and a list of explanations for each change.\nFormat:\n---\nYAML:\n<corrected_yaml>\n---\nJSON:\n<corrected_json>\n---\nEXPLANATIONS:\n- <explanation1>\n- <explanation2>\n...\n"""
    response = llm.invoke(prompt)
    yaml_part, json_part, explanations = '', '', []
    if 'YAML:' in response and 'JSON:' in response and 'EXPLANATIONS:' in response:
        try:
            yaml_part = response.split('YAML:')[1].split('---')[0].strip()
            json_part = response.split('JSON:')[1].split('---')[0].strip()
            explanations = [line.strip(
                '- ').strip() for line in response.split('EXPLANATIONS:')[1].split('\n') if line.strip()]
        except Exception:
            pass
    return {
        'yaml': yaml_part,
        'json': json_part,
        'explanations': explanations,
        'raw_response': response
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
