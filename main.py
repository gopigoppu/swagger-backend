import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import aiofiles
import tempfile
import shutil
import yaml
import json
from langgraph_flow import run_correction_pipeline, validate_openapi_spec, get_llm
import collections.abc

load_dotenv()

app = FastAPI(title="Swagger/OpenAPI Validator & Corrector")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = tempfile.gettempdir()


def make_json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    # For objects like AIMessage, fallback to str
    return str(obj)


@app.post("/upload")
async def upload(file: UploadFile = File(None), url: str = Form(None)):
    """
    Accepts a Swagger/OpenAPI file upload or a public URL. Stores and returns the content.
    """
    if file:
        content = (await file.read()).decode()
    elif url:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.text
    else:
        return JSONResponse({"error": "No file or URL provided."}, status_code=400)
    # Optionally save to disk
    temp_path = os.path.join(
        UPLOAD_DIR, f"openapi_{next(tempfile._get_candidate_names())}")
    async with aiofiles.open(temp_path, 'w') as f:
        await f.write(content)
    return {"content": content, "path": temp_path}


@app.post("/validate")
async def validate(request: Request):
    """
    Validates the uploaded OpenAPI spec. Returns structured errors.
    """
    data = await request.json()
    content = data.get("content")
    is_valid, errors = validate_openapi_spec(content)
    return {"valid": is_valid, "errors": errors}


@app.post("/llm-correct")
async def llm_correct(request: Request):
    """
    Runs the LangGraph LLM correction pipeline and streams corrections via SSE.
    """
    data = await request.json()
    content = data.get("content")

    def event_stream():
        # Step 1: Validate
        is_valid, errors = validate_openapi_spec(content)
        yield f"event: progress\ndata: {{\"step\": \"validate\", \"valid\": {str(is_valid).lower()}, \"errors\": {json.dumps(errors)} }}\n\n"
        if is_valid:
            yield f"event: done\ndata: {{\"valid\": true, \"corrected\": null, \"explanations\": []}}\n\n"
            return
        # Step 2: Correction
        result = run_correction_pipeline(content, llm=get_llm())
        safe_result = make_json_safe(result)
        yield f"event: correction\ndata: {json.dumps(safe_result)}\n\n"
        yield f"event: done\ndata: {json.dumps(safe_result)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate")
async def generate(request: Request):
    """
    Accepts a user description and uses LangGraph/LLM to generate a full OpenAPI spec.
    """
    data = await request.json()
    description = data.get("description")
    llm = get_llm()
    prompt = f"Generate a complete OpenAPI 3.0 spec (YAML and JSON) for the following API description:\n{description}\nReturn YAML and JSON."
    response = llm.invoke(prompt)
    # Parse YAML and JSON from response (simple split)
    yaml_part, json_part = '', ''
    if 'YAML:' in response and 'JSON:' in response:
        try:
            yaml_part = response.split('YAML:')[1].split('---')[0].strip()
            json_part = response.split('JSON:')[1].split('---')[0].strip()
        except Exception:
            pass
    return {"yaml": yaml_part, "json": json_part, "raw": response}
