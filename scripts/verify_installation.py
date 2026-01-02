#!/usr/bin/env python3
"""
7D mH-Q Installation Verification Script
Validates all system components are properly installed.
"""

import os
import sys
import platform
import subprocess
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

class SystemVerifier:
    """Comprehensive system verification for 7D mH-Q"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def check(self, name, condition, warning=False):
        """Record a check result"""
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        elif warning:
            print(f"  ⚠ {name}")
            self.warnings += 1
        else:
            print(f"  ✗ {name}")
            self.failed += 1
        return condition
    
    def verify_python(self):
        """Check Python environment"""
        print("\n🐍 PYTHON ENVIRONMENT")
        print("-" * 40)
        
        version = sys.version_info
        self.check(f"Python {version.major}.{version.minor}.{version.micro}", version >= (3, 8))
        self.check(f"Platform: {platform.system()}", True)
        self.check(f"Architecture: {platform.machine()}", True)
    
    def verify_dependencies(self):
        """Check required packages"""
        print("\n📦 DEPENDENCIES")
        print("-" * 40)
        
        required = {
            'numpy': 'Core math',
            'scipy': 'Scientific computing',
            'requests': 'HTTP client',
        }
        
        optional = {
            'cupy': 'CUDA acceleration',
            'fastapi': 'Dashboard server',
            'uvicorn': 'ASGI server',
            'torch': 'Deep learning',
        }
        
        for pkg, desc in required.items():
            try:
                mod = __import__(pkg)
                ver = getattr(mod, '__version__', 'unknown')
                self.check(f"{pkg} ({ver}): {desc}", True)
            except ImportError:
                self.check(f"{pkg}: {desc} [MISSING]", False)
        
        for pkg, desc in optional.items():
            try:
                mod = __import__(pkg)
                ver = getattr(mod, '__version__', 'unknown')
                self.check(f"{pkg} ({ver}): {desc}", True, warning=False)
            except ImportError:
                self.check(f"{pkg}: {desc} [optional]", False, warning=True)
    
    def verify_gpu(self):
        """Check GPU capabilities"""
        print("\n🎮 GPU DETECTION")
        print("-" * 40)
        
        # NVIDIA CUDA
        cuda_found = False
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    self.check(f"NVIDIA GPU: {line}", True)
                cuda_found = True
        except:
            pass
        
        # Check CUDA runtime
        try:
            import cupy as cp
            device = cp.cuda.Device(0)
            mem = device.mem_info
            self.check(f"CUDA Runtime: {mem[1]/1e9:.1f} GB VRAM", True)
        except:
            if cuda_found:
                self.check("CUDA Runtime: Not detected (install cupy)", False, warning=True)
        
        # AMD HIP
        try:
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.check("AMD ROCm/HIP: Available", True)
        except:
            self.check("AMD ROCm/HIP: Not detected", False, warning=True)
        
        if not cuda_found:
            self.check("GPU: Using CPU fallback", True, warning=True)
    
    def verify_crystal_core(self):
        """Verify Crystal Architecture components"""
        print("\n💎 CRYSTAL CORE")
        print("-" * 40)
        
        # Check core files
        core_files = [
            'crystal_patterns.py',
            'genesis.py',
            'sovereign_genesis.py',
            'autonomous_reasoner.py',
        ]
        
        for f in core_files:
            path = os.path.join(CRYSTAL_DIR, f)
            self.check(f"{f}", os.path.exists(path))
        
        # Test import
        try:
            from crystal_patterns import CrystalPatternGenerator
            gen = CrystalPatternGenerator()
            self.check("CrystalPatternGenerator: Loadable", True)
            
            # Quick manifold test
            manifold = gen.generate_holographic_manifold(7, 4)
            self.check(f"7D Manifold: {manifold.shape}", manifold.shape[-1] == 7)
            
        except Exception as e:
            self.check(f"Crystal Core: {e}", False)
    
    def verify_neural_core(self):
        """Verify Neural Core components"""
        print("\n🧠 NEURAL CORE")
        print("-" * 40)
        
        neural_files = [
            'neural_core/amd_entropy_miner.py',
        ]
        
        for f in neural_files:
            path = os.path.join(CRYSTAL_DIR, f)
            self.check(f"{f}", os.path.exists(path))
        
        try:
            from neural_core.amd_entropy_miner import CrystalEntropyMiner
            miner = CrystalEntropyMiner(complexity=64)
            seed = miner.mine()
            self.check(f"Entropy Miner: {seed.shape} seed", len(seed) == 64)
        except Exception as e:
            self.check(f"Entropy Miner: {e}", False, warning=True)
    
    def verify_holographic_bridge(self):
        """Verify Holographic Bridge"""
        print("\n🌉 HOLOGRAPHIC BRIDGE")
        print("-" * 40)
        
        bridge_files = [
            'holographic_bridge/__init__.py',
            'holographic_bridge/hip_launcher.py',
            'holographic_bridge/holographic_tensor.py',
        ]
        
        for f in bridge_files:
            path = os.path.join(CRYSTAL_DIR, f)
            self.check(f"{f}", os.path.exists(path))
    
    def verify_llm_connection(self):
        """Check LLM API connectivity"""
        print("\n🤖 LLM CONNECTIVITY")
        print("-" * 40)
        
        import requests
        
        endpoints = [
            ("http://127.0.0.1:1234/v1/models", "LM Studio (local)"),
            ("http://10.5.0.2:1234/v1/models", "LM Studio (network)"),
        ]
        
        any_connected = False
        for url, name in endpoints:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    self.check(f"{name}: Connected", True)
                    any_connected = True
            except:
                self.check(f"{name}: Offline", False, warning=True)
        
        if not any_connected:
            self.check("LLM: Running in autonomous mode", True, warning=True)
    
    def verify_file_integrity(self):
        """Check file integrity of key components"""
        print("\n🔐 FILE INTEGRITY")
        print("-" * 40)
        
        key_files = [
            'crystal_patterns.py',
            'genesis.py',
        ]
        
        for f in key_files:
            path = os.path.join(CRYSTAL_DIR, f)
            if os.path.exists(path):
                with open(path, 'rb') as fp:
                    content = fp.read()
                    hash_val = hashlib.md5(content).hexdigest()[:12]
                    size = len(content)
                    self.check(f"{f}: {size} bytes (hash: {hash_val})", True)
    
    def run_all(self):
        """Run complete verification"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║         7D mH-Q INSTALLATION VERIFICATION                    ║
║   Manifold-Constrained Holographic Quantum Architecture      ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        self.verify_python()
        self.verify_dependencies()
        self.verify_gpu()
        self.verify_crystal_core()
        self.verify_neural_core()
        self.verify_holographic_bridge()
        self.verify_llm_connection()
        self.verify_file_integrity()
        
        self.summary()
    
    def summary(self):
        """Print verification summary"""
        print("\n" + "="*60)
        print("  VERIFICATION SUMMARY")
        print("="*60)
        
        total = self.passed + self.failed + self.warnings
        print(f"  Passed:   {self.passed}/{total}")
        print(f"  Failed:   {self.failed}/{total}")
        print(f"  Warnings: {self.warnings}/{total}")
        
        if self.failed == 0:
            print("\n  ✅ SYSTEM READY")
        else:
            print(f"\n  ❌ {self.failed} CRITICAL ISSUES FOUND")
        
        print("="*60)

def main():
    verifier = SystemVerifier()
    verifier.run_all()

if __name__ == "__main__":
    main()
