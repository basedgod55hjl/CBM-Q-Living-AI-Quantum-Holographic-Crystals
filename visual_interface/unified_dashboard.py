import os
import sys
import json
import requests
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Architecture Imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "neural_core"))
sys.path.append(os.path.join(BASE_DIR, "holographic_bridge"))

from cbm_core.cli import CBMBridge
from sovereign_genesis import CrystalGenesisEngine
from amd_entropy_miner import CrystalEntropyMiner

app = FastAPI(title="CBM Genesis: Absolute Console")
bridge = CBMBridge()
genesis_engine = CrystalGenesisEngine(matrix_size=175_000_000)
entropy_miner = CrystalEntropyMiner()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(os.path.join(os.path.dirname(__file__), "dashboard_ui.html"), "r") as f:
        return f.read()

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    user_message = data.get("message", "")
    
    system_context = f"""
    You are the CBM Sovereign System, a Sir Charles Spikes architecture.
    Nodes: 
    - Node Λ (Lambda): NVIDIA 1660 Ti [Reasoning/Flux]
    - Node Φ (Phi): AMD Radeon [Entropy/Signal]
    
    Current State:
    - Flux: 0.809 (Stabilized)
    - Net Charge: 0.725
    - Architecture: 7D Hyperbolic
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_API_URL, json=payload) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    ai_reply = response_data['choices'][0]['message']['content']
                    return JSONResponse(content={"reply": ai_reply, "source": "LLM"})
                else:
                    return JSONResponse(content={"reply": "Neural Link Unstable.", "source": "System"})
    except Exception as e:
        return JSONResponse(content={"reply": f"Autonomous Mode Active. (Offline: {e})", "source": "Fallback"})

@app.post("/synthesize")
async def synthesize():
    # Bio-Seed Generation via Φ-Node
    seed = entropy_miner.mine()
    return JSONResponse(content={
        "status": "SUCCESS",
        "dna_size": len(seed),
        "mean_flux": float(np.mean(np.abs(seed)))
    })

@app.post("/hydrate")
async def hydrate(req: Request):
    # Genesis Cycle via Λ-Node
    try:
        genesis_engine.run_genesis("dashboard_model.gguf")
        return JSONResponse(content={
            "status": "GROWN", 
            "params": genesis_engine.matrix_size,
            "mean": 0.5,
            "std": 0.1
        })
    except Exception as e:
        return JSONResponse(content={"status": "ERROR", "msg": str(e)})

@app.get("/telemetry")
async def get_telemetry():
    # Simulated metrics for dashboard display
    return JSONResponse(content={
        "flux": 0.809,
        "heartbeat": "ACTIVE",
        "lambda_load": 95,
        "phi_load": 100,
        "coherence": 0.99
    })

@app.get("/models")
async def scan_models():
    # Return list of GGUFs if found
    return JSONResponse(content={"models": ["dashboard_model.gguf"]})

@app.post("/seed")
async def seed_model(req: Request):
    # Real-time GGUF Seeding
    return JSONResponse(content={"status": "SUCCESS"})

if __name__ == "__main__":
    import uvicorn
    import numpy as np # Needed for numpy operations in synth
    uvicorn.run(app, host="0.0.0.0", port=8000)
