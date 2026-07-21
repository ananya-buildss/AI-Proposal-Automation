from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
from datetime import datetime

from services.gemini_service import (
    generate_proposal,
    classify_document
)
from services.ml_service import analyze_requirement
from services.rag_service import retrieve_similar_projects
from services.export_service import export_proposal
from services.pdf_export_service import export_pdf

from utils.pdf_reader import extract_text_from_pdf
from prompts.proposal_prompt import build_prompt

app = FastAPI(title="AI Proposal Automation API")

# ----------------------------------------------------
# CORS
# ----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Static Files
# ----------------------------------------------------

os.makedirs("generated_proposals", exist_ok=True)

app.mount(
    "/generated_proposals",
    StaticFiles(directory="generated_proposals"),
    name="generated_proposals"
)

# ----------------------------------------------------
# Models
# ----------------------------------------------------

class Requirement(BaseModel):
    text: str


# ----------------------------------------------------
# AI Pipeline
# ----------------------------------------------------

def process_requirement(requirement_text: str):

    analysis = analyze_requirement(requirement_text)

    similar_projects = retrieve_similar_projects(requirement_text)

    prompt = build_prompt(
        requirement=requirement_text,
        analysis=analysis,
        similar_projects=similar_projects
    )

    proposal = generate_proposal(prompt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"proposal_{timestamp}"

    export_proposal(
        proposal,
        filename
    )

    export_pdf(
        proposal,
        filename
    )

    return {
        "analysis": analysis,
        "similar_projects": similar_projects,
        "proposal": proposal,
        "docx_download": f"generated_proposals/{filename}.docx",
        "pdf_download": f"generated_proposals/{filename}.pdf"
    }


# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Proposal Automation Backend Running"
    }


@app.post("/generate")
def generate(req: Requirement):
    return process_requirement(req.text)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # --------------------------------------------
        # Extract Text
        # --------------------------------------------

        extracted_text = extract_text_from_pdf(file_path)

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Unable to extract text from the uploaded PDF."
            )

        # --------------------------------------------
        # Gemini Document Validation
        # --------------------------------------------

        document_type = classify_document(extracted_text).strip().lower()

        if "requirement" not in document_type:
            raise HTTPException(
                status_code=400,
                detail="Invalid document. Please upload a valid software requirement document."
            )

        # --------------------------------------------
        # Generate Proposal
        # --------------------------------------------

        result = process_requirement(extracted_text)
        result["filename"] = file.filename

        return result

    finally:
        # Delete uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)


# ----------------------------------------------------
# Download Route
# ----------------------------------------------------

@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(
        "generated_proposals",
        filename
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found."
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )