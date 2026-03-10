"""
FastAPI application (simplified for flat structure)
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

app = FastAPI(
    title="Multimodal Medical Assistant API",
    description="AI-powered medical assistant for processing text and images",
    version="1.0.0"
)


class TextInput(BaseModel):
    """Text input model"""
    text: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Medical Assistant API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze_text": "/api/analyze-text",
            "analyze_image": "/api/analyze-image",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multimodal-medical-assistant"}


@app.post("/api/analyze-text")
async def analyze_text(input_data: TextInput):
    """
    Analyze medical text.
    
    Args:
        input_data: Text input
        
    Returns:
        Analysis results
    """
    try:
        return {
            "prediction": "Normal",
            "confidence": 0.85,
            "text_length": len(input_data.text),
            "note": "Demo mode - using placeholder predictions"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze medical image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Analysis results
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        return {
            "prediction": "Normal",
            "confidence": 0.82,
            "image_size": image.size,
            "note": "Demo mode - using placeholder predictions"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
