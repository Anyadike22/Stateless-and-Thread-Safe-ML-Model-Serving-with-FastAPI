# Stateless-and-Thread-Safe-ML-Model-Serving-with-FastAPI




























```python
from fastapi import FastAPI, Depends
from functools import lru_cache
import pickle

app = FastAPI()

# ----------------------------
# Stateless ModelPredictor Class
# ----------------------------
class ModelPredictor:
    def __init__(self, model):
        # Model is loaded once and treated as immutable (read-only)
        self.model = model  # No internal state changes allowed!

    def preprocess(self, input_data: dict) -> list:
        # Stateless preprocessing: No side effects or mutable variables.
        # Example: Validate and transform input (pure function logic).
        return [input_data["feature_1"], input_data["feature_2"]]

    def predict(self, processed_data: list) -> float:
        # Stateless prediction: Uses only inputs and the immutable model.
        return self.model.predict([processed_data])[0]

# ----------------------------
# Dependency Injection (Load Model Once)
# ----------------------------
@lru_cache
def load_model():
    # Load model at startup (immutable and thread-safe if model is stateless)
    with open("model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    return ModelPredictor(model)  # Class instance with immutable model

# ----------------------------
# FastAPI Endpoint
# ----------------------------
@app.post("/predict")
async def predict_endpoint(
    data: dict, 
    predictor: ModelPredictor = Depends(load_model)  # Injects stateless instance
):
    processed_data = predictor.preprocess(data)
    prediction = predictor.predict(processed_data)
    return {"prediction": prediction}
```







    
