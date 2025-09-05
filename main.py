import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import warnings
import io
import re

# --- Settings ---
warnings.filterwarnings('ignore')

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Customer Segmentation API",
    description="Upload a CSV, segment customers, and store them in Supabase."
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define the Mapping to Your Exact Supabase Table Names ---
TABLE_NAME_MAP = {
    '1. New & Cautious': 'new_and_cautious',
    '2. Stable Earners': 'stable_earners',
    '3. Mid-Tier Professionals': 'mid_tier_professionals',
    '4. Affluent Customers': 'affluent_customers',
    '5. High-Value Elite': 'high_value_elite'
}

# --- Load Model and Connect to Supabase ---
try:
    model_pipeline = joblib.load('customer_segmentation_model.joblib')
    print("✅ Trained model loaded successfully.")

    SUPABASE_URL = "https://gryogruqtbbobildxrcz.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdyeW9ncnVxdGJib2JpbGR4cmN6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjgzMDM5MCwiZXhwIjoyMDcyNDA2MzkwfQ.C_AD06xF9No-yWV1grkDaNceBtLNtPBEORCDVI7ZiiU"
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase client initialized.")

except FileNotFoundError:
    print("❌ FATAL ERROR: 'customer_segmentation_model.joblib' not found.")
    model_pipeline = None

# --- API Endpoint ---
@app.post("/segment-and-store")
async def segment_and_store(file: UploadFile = File(...)):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        contents = await file.read()
        df_new = pd.read_csv(io.BytesIO(contents))
        
        data_to_segment = df_new.copy()
        data_to_segment.replace(['_INVALID_', '_RARE_'], np.nan, inplace=True)
        data_to_segment.dropna(inplace=True)
        
        for col in ['has_loan', 'has_credit_card', 'has_investment']:
            if col in data_to_segment.columns:
                data_to_segment[col] = data_to_segment[col].astype(int)
        for col in ['age', 'income', 'balance', 'account_tenure']:
            if col in data_to_segment.columns:
                data_to_segment[col] = pd.to_numeric(data_to_segment[col], errors='coerce')
        data_to_segment.dropna(inplace=True)

        predicted_clusters = model_pipeline.predict(data_to_segment)
        data_to_segment['cluster'] = predicted_clusters
        
        print(f"✅ Successfully segmented {len(data_to_segment)} customers.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    # --- Data-Driven Naming Logic ---
    cluster_summary = data_to_segment.groupby('cluster')[['income', 'balance', 'age']].mean()
    cluster_summary.sort_values(by=['income', 'balance'], inplace=True)
    
    sorted_persona_names = list(TABLE_NAME_MAP.keys())
    
    final_label_map = {cluster_id: name for cluster_id, name in zip(cluster_summary.index, sorted_persona_names)}
    data_to_segment['persona_name'] = data_to_segment['cluster'].map(final_label_map)
    print("✅ Persona names assigned.")

    # --- Store Data in Supabase ---
    results = {"message": "Segmentation successful.", "clusters": {}}
    
    for persona_name in sorted_persona_names:
        table_name = TABLE_NAME_MAP.get(persona_name)
        if not table_name:
            continue

        cluster_df = data_to_segment[data_to_segment['persona_name'] == persona_name]
        
        if not cluster_df.empty:
            records = cluster_df.drop(columns=['cluster', 'persona_name']).to_dict(orient='records')
            try:
                # --- (FIX) Use .upsert() instead of .insert() ---
                # This will update existing records and insert new ones.
                data, count = supabase.table(table_name).upsert(records, on_conflict='customer_id').execute()
                
                results["clusters"][table_name] = {"processed_count": len(records), "status": "success"}
                print(f"  -> Upserted {len(records)} records into {table_name}")
            except Exception as e:
                results["clusters"][table_name] = {"processed_count": 0, "status": "failed", "error": str(e)}
                print(f"  -> FAILED to upsert into {table_name}: {e}")

    return results

# To run the server: uvicorn main:app --reload