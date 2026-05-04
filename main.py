from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import DATA_DIR
from forms import extract_form, list_schemas, load_schema
from ingest import build_index, load_index
from rag import generate, retrieve
from lessons import lessons_summary, list_reasons, clear_all_lessons
from worklist import (
    clear_all,
    compute_stats,
    delete_record,
    get_record,
    list_records,
    list_reviewers,
    save_extraction,
    seed_mock_records,
    update_record,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory state
faiss_index = None
faiss_metadata = None


@app.on_event("startup")
def startup():
    global faiss_index, faiss_metadata
    result = load_index()
    if result:
        faiss_index, faiss_metadata = result


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        dest = DATA_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append(file.filename)

    if saved:
        build_index()
        global faiss_index, faiss_metadata
        result = load_index()
        if result:
            faiss_index, faiss_metadata = result

    return {"uploaded": saved}


@app.get("/documents")
async def documents():
    pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    return {"documents": pdfs}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = DATA_DIR / filename
    if file_path.exists():
        file_path.unlink()
    build_index()
    global faiss_index, faiss_metadata
    result = load_index()
    if result:
        faiss_index, faiss_metadata = result
    else:
        faiss_index, faiss_metadata = None, None
    return {"deleted": filename}


@app.get("/forms/schemas")
async def forms_schemas():
    return {"schemas": list_schemas()}


@app.get("/forms/schemas/{form_id}")
async def forms_schema_detail(form_id: str):
    return load_schema(form_id)


@app.post("/forms/extract")
async def forms_extract(form_id: str, file: UploadFile = File(...)):
    content = await file.read()
    result = await extract_form(content, file.filename, form_id)
    if "error" in result:
        return result
    record = save_extraction(result, file.filename)
    result["record_id"] = record["id"]
    result["status"] = record["status"]
    result["assigned_to"] = record["assigned_to"]
    result["assigned_to_name"] = record["assigned_to_name"]
    return result


@app.get("/worklist")
async def worklist(status: str | None = None, assigned_to: str | None = None,
                    date_from: str | None = None, date_to: str | None = None):
    records = list_records(status=status, assigned_to=assigned_to,
                            date_from=date_from, date_to=date_to)
    # Trim heavy fields for the list view
    summaries = []
    for r in records:
        summaries.append({
            "id": r["id"],
            "filename": r["filename"],
            "form_id": r["form_id"],
            "form_name": r["form_name"],
            "extracted_at": r["extracted_at"],
            "status": r["status"],
            "assigned_to": r["assigned_to"],
            "assigned_to_name": r["assigned_to_name"],
            "overall_confidence": r["overall_confidence"],
            "overall_trust_score": r.get("overall_trust_score", r["overall_confidence"]),
            "hallucination_catches": r.get("hallucination_catches", 0),
            "cross_field_issues_count": r.get("cross_field_issues_count", 0),
            "flagged_count": len(r.get("flagged_fields", [])),
        })
    return {"records": summaries}


@app.get("/worklist/reviewers")
async def worklist_reviewers():
    return {"reviewers": list_reviewers()}


@app.get("/worklist/stats")
async def worklist_stats(date_from: str | None = None, date_to: str | None = None):
    return compute_stats(date_from=date_from, date_to=date_to)


@app.post("/worklist/seed")
async def worklist_seed(count: int = 18):
    created = seed_mock_records(count)
    return {"created": created}


@app.post("/worklist/clear")
async def worklist_clear():
    deleted = clear_all()
    return {"deleted": deleted}


@app.get("/worklist/{record_id}")
async def worklist_detail(record_id: str):
    record = get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.get("/lessons/reasons")
async def lessons_reasons():
    return {"reasons": list_reasons()}


@app.get("/lessons/stats")
async def lessons_stats(form_id: str | None = None):
    return lessons_summary(form_id=form_id)


@app.post("/lessons/clear")
async def lessons_clear():
    return {"deleted": clear_all_lessons()}


@app.put("/worklist/{record_id}")
async def worklist_update(record_id: str, payload: dict = Body(...)):
    record = update_record(
        record_id,
        edits=payload.get("edits"),
        edit_reasons=payload.get("edit_reasons"),
        status=payload.get("status"),
        assigned_to=payload.get("assigned_to"),
        actor=payload.get("actor", "reviewer"),
    )
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/worklist/{record_id}")
async def worklist_delete(record_id: str):
    deleted = delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"deleted": record_id}


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    history = body.get("history", [])

    context_chunks = []
    if faiss_index is not None and faiss_metadata is not None:
        context_chunks = retrieve(query, faiss_index, faiss_metadata)

    sources = [{"source_file": c["source_file"], "score": c["score"]} for c in context_chunks]

    async def event_stream():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        # Stream tokens
        async for token in generate(query, context_chunks, history):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
