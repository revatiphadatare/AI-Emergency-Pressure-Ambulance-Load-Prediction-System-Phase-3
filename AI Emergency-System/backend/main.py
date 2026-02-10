from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from ml.model import predict_load

app = FastAPI(title="AI Emergency Prediction API")

# Controlled CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(
    city: str = Query(..., min_length=2),
    hour: int = Query(..., ge=0, le=23),
    emergency_calls: int = Query(..., ge=0, le=500)
):
    try:
        return predict_load(city, hour, emergency_calls)
    except Exception:
        return {"error": "Prediction failed. Please try again."}
