import os
import sys
import webbrowser
import uvicorn
import threading
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# 1. Setup Architecture Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_DIR = os.path.join(BASE_DIR, "holographic_bridge")
CORE_DIR = os.path.join(BASE_DIR, "neural_core")
UI_DIR = os.path.join(BASE_DIR, "visual_interface")

sys.path.append(BASE_DIR)

print("[*] HOLOGRAPHIC AI CRYSTALS ARCHITECTURE: INITIALIZING...")
print(f"   [+] Bridge: {BRIDGE_DIR}")
print(f"   [+] Core:   {CORE_DIR}")
print(f"   [+] UI:     {UI_DIR}")

# 2. Initialize Core Logic
try:
    from cbm_core.cli import CBMBridge
    from cbm_core.cbm_rust_core import PyWASMOrchestrator
    import requests
    
    print("   [+] Rust Core: DETECTED")
    bridge = CBMBridge()
    # Verify mH-QA Stability & Sacred Geometry
    axioms = ["Genesis", "Launch", "mH-QA"]
    seed_res = bridge.synthesize_seed(axioms)
    if seed_res.get("pattern") == "FibonacciSpiral":
        print("   [+] Sacred Geometry: VERIFIED")
        print("   [+] mH-QA Manifold Stability: COHERENT")
    else:
        print("   [!] Sacred Geometry: UNVERIFIED")
        print("   [!] mH-QA Stability Check: DEGRADED")

    # Verify Neural Link (LLM)
    LLM_API_URL = "http://127.0.0.1:1234/v1/models" # Check models endpoint for health
    try:
        response = requests.get(LLM_API_URL, timeout=2)
        if response.status_code == 200:
            print("   [+] Neural Link (LM Studio): CONNECTED")
        else:
            print(f"   [!] Neural Link: UNSTABLE (Status {response.status_code})")
            print("   [!] SYSTEM: Running in Autonomous Fallback Mode")
    except:
        print("   [!] Neural Link: OFFLINE")
        print("   [!] SYSTEM: Running in Autonomous Fallback Mode")

except ImportError as e:
    print(f"   [!] CRITICAL: Core Modules Missing - {e}")
    sys.exit(1)

# 3. Setup Interface
from visual_interface.unified_dashboard import app as dashboard_app

def launch_server():
    uvicorn.run(dashboard_app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    print("\n[LAUNCH] VISUAL INTERFACE...")
    
    # Start Server Thread
    server_thread = threading.Thread(target=launch_server, daemon=True)
    server_thread.start()
    
    # Wait for startup
    time.sleep(2)
    
    # Open Browser
    print("   [+] Opening Crystal Holographic Dashboard...")
    webbrowser.open("http://127.0.0.1:8000")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] SYSTEM...")
