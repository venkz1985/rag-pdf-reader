import json
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import DATA_DIR
from ingest import build_index, load_index
from rag import generate, retrieve

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
    return templates.TemplateResponse("index.html", {"request": request})


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
