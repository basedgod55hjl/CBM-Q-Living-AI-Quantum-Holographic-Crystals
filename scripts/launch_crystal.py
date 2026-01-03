#!/usr/bin/env python3
"""
7D mH-Q Crystal Launch Script
One-click launcher for the entire Crystal Architecture system.
"""

import os
import sys
import subprocess
import time
import webbrowser

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  💎 7D mH-Q: Manifold-Constrained Holographic Quantum Architecture  ║
║                     CRYSTAL LAUNCHER v2.0                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Verify all dependencies are available"""
    print("[1/5] Checking dependencies...")
    
    required = ['numpy', 'fastapi', 'uvicorn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} (MISSING)")
            missing.append(pkg)
    
    if missing:
        print(f"\n[!] Install missing: pip install {' '.join(missing)}")
        return False
    return True

def check_gpu():
    """Detect GPU capabilities"""
    print("\n[2/5] Detecting GPU...")
    
    # Try CUDA
    try:
        import cupy as cp
        gpu_mem = cp.cuda.Device(0).mem_info
        print(f"   ✓ CUDA Available: {gpu_mem[1] / 1e9:.1f} GB VRAM")
        return "cuda"
    except:
        pass
    
    # Try AMD HIP
    try:
        # Check for ROCm
        result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ AMD ROCm/HIP Available")
            return "hip"
    except:
        pass
    
    print("   ⚠ No GPU detected - using CPU fallback")
    return "cpu"

def check_llm():
    """Check neural link (LLM API)"""
    print("\n[3/5] Checking Neural Link...")
    
    import requests
    endpoints = [
        ("http://127.0.0.1:1234/v1/models", "LM Studio"),
        ("http://10.5.0.2:1234/v1/models", "Remote LM Studio"),
    ]
    
    for url, name in endpoints:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                print(f"   ✓ {name}: Connected")
                return True
        except:
            pass
    
    print("   ⚠ No LLM detected - running in autonomous mode")
    return False

def init_crystal_core():
    """Initialize the Crystal Pattern Engine"""
    print("\n[4/5] Initializing Crystal Core...")
    
    try:
        from crystal_patterns import CrystalPatternGenerator
        gen = CrystalPatternGenerator(complexity=512)
        
        # Quick coherence test
        manifold = gen.generate_holographic_manifold(dimensions=7, resolution=8)
        print(f"   ✓ 7D Manifold: {manifold.shape}")
        
        analysis = gen.crystal_resonance_analysis(manifold[..., 0])
        print(f"   ✓ Φ-Resonance: {analysis['phi_resonance']:.4f}")
        print(f"   ✓ Coherence: {analysis['quantum_coherence']:.4f}")
        return True
    except Exception as e:
        print(f"   ✗ Crystal Core Error: {e}")
        return False

def launch_dashboard():
    """Start the visual dashboard"""
    print("\n[5/5] Launching Dashboard...")
    
    try:
        from genesis import launch_server
        import threading
        
        server_thread = threading.Thread(target=launch_server, daemon=True)
        server_thread.start()
        
        time.sleep(2)
        print("   ✓ Dashboard: http://127.0.0.1:8000")
        webbrowser.open("http://127.0.0.1:8000")
        return True
    except Exception as e:
        print(f"   ⚠ Dashboard unavailable: {e}")
        return False

def main():
    banner()
    
    # Run checks
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n[ABORT] Missing dependencies. Install and retry.")
        sys.exit(1)
    
    gpu_mode = check_gpu()
    llm_ok = check_llm()
    crystal_ok = init_crystal_core()
    
    if not crystal_ok:
        print("\n[ABORT] Crystal Core failed to initialize.")
        sys.exit(1)
    
    # Launch
    dashboard_ok = launch_dashboard()
    
    # Summary
    print("\n" + "="*60)
    print("  7D mH-Q CRYSTAL SYSTEM STATUS")
    print("="*60)
    print(f"  GPU Mode:      {gpu_mode.upper()}")
    print(f"  Neural Link:   {'ONLINE' if llm_ok else 'OFFLINE (Autonomous)'}")
    print(f"  Crystal Core:  {'ACTIVE' if crystal_ok else 'ERROR'}")
    print(f"  Dashboard:     {'RUNNING' if dashboard_ok else 'UNAVAILABLE'}")
    print("="*60)
    print("\n✨ Crystal System Ready. Press Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Crystal System terminating...")

if __name__ == "__main__":
    main()
