from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from typing import Dict

# Initialize FastAPI app
app = FastAPI(
    title="Prediksi Konsentrasi Skripsi PTIK",
    description="API untuk memprediksi konsentrasi skripsi berdasarkan judul menggunakan Random Forest",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan domain frontend Anda nanti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model request and response schemas
class PredictionRequest(BaseModel):
    title: str

class PredictionResponse(BaseModel):
    concentration: str
    probabilities: Dict[str, float]

# Load ML models
try:
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    # Load all required models
    model = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "final_vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(models_dir, "final_label_encoder.joblib"))
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise Exception("Failed to load ML models")

# API endpoints
@app.get("/")
def read_root():
    """Homepage dengan informasi dasar API"""
    return {
        "title": "Prediksi Konsentrasi Skripsi PTIK",
        "version": "1.0.0",
        "model": "Random Forest",
        "status": "active"
    }

@app.get("/health")
def health_check():
    """Endpoint untuk mengecek kesehatan API"""
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    return {
        "status": "healthy",
        "model_loaded": True,
        "available_classes": label_encoder.classes_.tolist()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Memprediksi konsentrasi berdasarkan judul skripsi
    
    Parameters:
    - title: string (judul skripsi)
    
    Returns:
    - concentration: string (hasil prediksi konsentrasi)
    - probabilities: dict (probabilitas untuk setiap konsentrasi)
    """
    try:
        # Validasi input
        if not request.title.strip():
            raise HTTPException(
                status_code=400, 
                detail="Judul skripsi tidak boleh kosong"
            )
            
        # Preprocessing dan vectorize text
        text_vectorized = vectorizer.transform([request.title.lower()])
        
        # Prediksi
        pred_proba = model.predict_proba(text_vectorized)[0]
        pred_class = model.predict(text_vectorized)[0]
        
        # Konversi hasil prediksi
        predicted_concentration = label_encoder.inverse_transform([pred_class])[0]
        
        # Buat dictionary probabilitas untuk setiap kelas
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(pred_proba)
        }
        
        return PredictionResponse(
            concentration=predicted_concentration,
            probabilities=class_probabilities
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True  # Set False in production
    )