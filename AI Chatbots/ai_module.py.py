from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import PyPDF2
from docx import Document
import json
from io import BytesIO
from typing import Dict, Any
import uvicorn

# Load environment variables
load_dotenv("my.env")

# Initialize FastAPI app
app = FastAPI(title="Resume Analyzer API", version="1.0.0")

# Set up Groq API
os.environ["GROQ_API_KEY"] = os.getenv("Groq")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# Define the prompt template
summary_prompt = PromptTemplate(
    template="""You are an expert resume reviewer. Your job is to analyze the following resume content and identify areas of improvement.

DOCUMENT CONTENT:
{text}

INSTRUCTIONS:
First, analyze the resume and identify  key improvement points only which are critical to be reported.
Second, return the improvements strictly in valid JSON format with keys as numbers (e.g., "1", "2", "3").
Third, do NOT include any extra text, explanations, or code blocks (like ```json). Only return the JSON object.
Fourth, follow this exact example format:
{{"1": "add projects", "2": "add experience", "3": "improve grammar"}}

OUTPUT:""",
    input_variables=["text"]
)


# Create the chain
summary_chain = summary_prompt | llm | StrOutputParser()

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        docx_file = BytesIO(file_content)
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def parse_json_response(response: str) -> Dict[Any, Any]:
    """Parse the JSON response from the LLM."""
    try:
        # Try to extract JSON if it's embedded in text
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire response
            return json.loads(response)
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw response
        return {"raw_response": response, "note": "Response was not in valid JSON format"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Resume Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload and analyze resume file (PDF or DOCX)"
        }
    }

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    """
    Analyze a resume file and provide improvement suggestions.
    
    Accepts PDF and DOCX files.
    Returns JSON with improvement suggestions.
    """
    
    # Check file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = file.filename.lower().split('.')[-1]
    
    if file_extension not in ['pdf', 'docx']:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload a PDF or DOCX file."
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_content)
        
        # Check if text was extracted
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        # Process with LLM
        result = summary_chain.invoke({"text": text})
        
        # Parse the JSON response
        parsed_result = parse_json_response(result)
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "file_type": file_extension,
            "improvements": parsed_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-text")
async def analyze_text(text_data: dict):
    """
    Analyze resume text directly without file upload.
    
    Request body should contain: {"text": "resume content here"}
    """
    
    if "text" not in text_data or not text_data["text"].strip():
        raise HTTPException(status_code=400, detail="No text provided or text is empty")
    
    try:
        # Process with LLM
        result = summary_chain.invoke({"text": text_data["text"]})
        
        # Parse the JSON response
        parsed_result = parse_json_response(result)
        
        return JSONResponse(content={
            "status": "success",
            "improvements": parsed_result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)