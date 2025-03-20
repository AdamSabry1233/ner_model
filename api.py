import spacy
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Correct Absolute Path to Your Model
MODEL_PATH = r"C:/Users/asabry/Desktop/Ner_model/ner_expense_model"

# Load the trained NER model
nlp = spacy.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class TextRequest(BaseModel):
    text: str

# API endpoint
@app.post("/extract-expenses/")
def extract_expenses(request: TextRequest):
    doc = nlp(request.text)
    extracted_data = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"extracted_entities": extracted_data}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
