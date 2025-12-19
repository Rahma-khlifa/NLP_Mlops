import os
import io
import json
import pickle
from typing import List, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

app = FastAPI(title="Sentiment API", version="1.0.0")

model = None
model_metadata = {}
vectorizer = None


def load_local_model():
    global model, model_metadata, vectorizer
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Try Docker volume path first, then local path
        model_paths = [
            '/model_registry/Best_Election_Model',  # Docker volume
            os.path.join(root, 'model_registry', 'Best_Election_Model'),  # Local
            os.path.join(root, 'notebooks', 'model_registry', 'Best_Election_Model')  # Legacy
        ]
        
        model_dir = None
        for path in model_paths:
            if os.path.exists(path):
                model_dir = path
                print(f'✅ Found model directory: {model_dir}')
                break
        
        if not model_dir:
            print(f'❌ Model directory not found in any of: {model_paths}')
            return
        
        # Load model
        prod_path = os.path.join(model_dir, 'production.pkl')
        if os.path.exists(prod_path):
            with open(prod_path, 'rb') as f:
                model = pickle.load(f)
            print(f'✅ Model loaded from: {prod_path}')
        else:
            print(f'⚠️  production.pkl not found at: {prod_path}')
        
        # Load vectorizer - try multiple locations
        vectorizer_paths = [
            os.path.join(model_dir, 'tfidf_vectorizer.pkl'),  # Same dir as model
            '/model_registry/Best_Election_Model/tfidf_vectorizer.pkl',  # Docker
            os.path.join(root, 'processors', 'tfidf_vectorizer.pkl'),  # Legacy
        ]
        
        for vec_path in vectorizer_paths:
            if os.path.exists(vec_path):
                with open(vec_path, 'rb') as vf:
                    vectorizer = pickle.load(vf)
                print(f'✅ Vectorizer loaded from: {vec_path}')
                break
        
        if vectorizer is None:
            print(f'⚠️  Vectorizer not found in any of: {vectorizer_paths}')
        
        # Load metadata if present
        versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if versions:
            latest = sorted(versions)[-1]
            meta_path = os.path.join(model_dir, latest, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    model_metadata = json.load(f)
                print(f'✅ Metadata loaded from: {meta_path}')
        
        print('✅ Model load attempt finished')
    except Exception as e:
        print(f'❌ Could not load model: {e}')
        import traceback
        traceback.print_exc()


@app.on_event('startup')
async def startup():
    load_local_model()


class TextInput(BaseModel):
    text: str


@app.get('/')
def root():
    return {'message': 'Sentiment API. Use /predict or /predict_csv'}


@app.get('/health')
def health():
    return {'status': 'ok' if model is not None else 'degraded', 'model_loaded': model is not None}


@app.post('/predict')
def predict(request: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    # Require a vectorizer to convert text -> numeric features for the saved SVC
    if vectorizer is None:
        raise HTTPException(status_code=503, detail='Vectorizer not available for inference')
    try:
        # Transform text to feature vector
        X = vectorizer.transform([request.text])
        # SVC expects a dense numeric array
        if hasattr(X, 'toarray'):
            X = X.toarray()
        preds = model.predict(X)
        return {'predictions': preds.tolist() if hasattr(preds, 'tolist') else preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict_csv')
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    if vectorizer is None:
        raise HTTPException(status_code=503, detail='Vectorizer not available for inference')
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='File must be CSV')
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail='CSV must contain a `text` column')

        # ✅ Transform text using the loaded TF-IDF vectorizer
        X = vectorizer.transform(df['text'].tolist())
        if hasattr(X, 'toarray'):
            X = X.toarray()
        preds = model.predict(X)

        df['predict'] = preds
        out = io.StringIO()
        df.to_csv(out, index=False)
        out.seek(0)
        return StreamingResponse(
            iter([out.getvalue()]), 
            media_type='text/csv', 
            headers={'Content-Disposition': f'attachment; filename=predictions_{file.filename}'}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
