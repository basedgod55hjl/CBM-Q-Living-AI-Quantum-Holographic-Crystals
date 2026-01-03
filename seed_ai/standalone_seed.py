import json
import time
import math
import random
import os
import sys
import numpy as np
from colorama import init, Fore, Style

init()

init()

import sys

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Go up from seed_ai/ to Crystal_Architecture/
DEFAULT_SEED = os.path.join(PROJECT_ROOT, "seeds", "genesis_seed.json")

# Add Project Root to Path for imports
sys.path.append(PROJECT_ROOT)

try:
    from crystal_patterns import CrystalPatternGenerator
    HAS_PATTERNS = True
except ImportError:
    HAS_PATTERNS = False

class CrystalSeedEntity:
    def __init__(self, seed_path):
        self.path = seed_path
        self.data = self._load()
        # Fallback for different seed schemas
        self.name = self.data.get('name', 'Unknown Seed')
        self.version = self.data.get('version', '1.0.0')
        self.arch = self.data.get('architecture', 'Crystal Logic')
        
        # Try to find PHI
        if 'sacred_constants' in self.data:
            self.phi = self.data['sacred_constants'].get('PHI', 1.618)
        elif 'parameters' in self.data:
            self.phi = self.data['parameters'].get('phi_weight', 1.618)
        else:
            self.phi = 1.6180339887
            
        self.state = 0.0
        
        # Initialize Real Pattern Generator
        if HAS_PATTERNS:
            self.generator = CrystalPatternGenerator(complexity=512)
        else:
            self.generator = None

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Seed not found at {self.path}")
        with open(self.path, 'r') as f:
            return json.load(f)

    def think(self, user_input):
        """
        Simulate 'thinking' using Crystal Logic (Algorithmic).
        Now upgraded with REAL Crystal Pattern Generation.
        """
        # Update internal state (Energy Accumulation)
        input_energy = sum(ord(c) for c in user_input)
        self.state += input_energy * self.phi
        
        # Parse Intent using Advanced Crystal Logic
        ui = user_input.lower().strip()
        parts = ui.split()
        cmd = parts[0] if parts else ""

        # 1. ENCODING LOGIC
        if cmd in ["encode", "hash"]:
            return self._response("ENCODE", context=" ".join(parts[1:]))

        # 2. SEQUENCE LOGIC (Fibonacci/Primes)
        elif any(x in ui for x in ["fib", "sequence", "series"]):
            return self._response("SEQUENCE")
        
        # 3. TEMPORAL LOGIC
        elif any(x in ui for x in ["time", "now", "date"]):
            return self._response("TIME")

        # 4. DIAGNOSTIC LOGIC
        elif any(x in ui for x in ["status", "health", "diag"]):
            return self._response("STATUS")

        # 5. PATTERN GENERATION (Real Crystal Logic)
        elif cmd in ["generate", "gen", "pattern"] or any(x in ui for x in ["geometry", "crystal", "fractal"]):
            return self._response("GENERATION", context=ui)
            
        # 6. QUANTUM EVOLUTION
        elif any(x in ui for x in ["evolve", "omega", "automata"]):
            return self._response("EVOLUTION", context=ui)

        # 7. HOLOGRAPHIC INTERFERENCE
        elif any(x in ui for x in ["interfere", "hologram", "wave"]):
            return self._response("INTERFERENCE", context=ui)

        # 8. CONVERSATIONAL LOGIC
        elif any(x in ui for x in ["hello", "hi", "greet"]):
            return self._response("GREETING")
        elif "who" in ui:
            return self._response("IDENTITY")
        elif any(x in ui for x in ["calc", "math", "state"]):
            return self._response("CALCULATION")
        elif "features" in ui:
            return self._response("FEATURES")
        
        # Default: Absorb Energy
        else:
            return self._response("UNKNOWN")

    def _response(self, intent, context=""):
        if intent == "GREETING":
            return f"Resonance established. I am {self.name}."
            
        elif intent == "IDENTITY":
            comp = self.data.get('complexity', 
                   self.data.get('parameters', {}).get('complexity', 'Unknown'))
            return f"I am a {self.arch} Entity. Complexity: {comp}. My harmonic base is {self.phi}."
            
        elif intent == "ENCODE":
            # Holographic Encoding Simulation
            if not context: return "Provide data to encode."
            chars = [ord(c) for c in context]
            # Phi-Interaction
            encoded = "".join([f"{int(c * self.phi):X}" for c in chars])
            return f"Holographic Cypher: 0x{encoded}"

        elif intent == "SEQUENCE":
            # Generate Fibonacci adjusted by Phi State
            seq = [1, 1]
            for _ in range(8):
                seq.append(seq[-1] + seq[-2])
            return f"Vital Sequence Generation: {seq} -> Converging to PHI."

        elif intent == "TIME":
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            return f"Temporal Anchor: {now} (Local Frame)"

        elif intent == "STATUS":
            stability = max(0, min(100, (math.sin(self.state) + 1) * 50))
            return f"System Stable. Energy State: {self.state:.2f}. Integrity: {stability:.1f}%"

        elif intent == "GENERATION":
            if not self.generator:
                return "Pattern Generator implementation missing (crystal_patterns.py not found)."
            
            # Determining pattern type
            if "flower" in context:
                pat = self.generator.generate_sacred_geometry("flower_of_life")
                info = f"Flower of Life generated. {len(pat['centers'])} circles."
            elif "metatron" in context:
                pat = self.generator.generate_metatron_cube()
                info = f"Metatron's Cube generated. {len(pat['vertices'])} vertices."
            elif "spiral" in context:
                pat = self.generator.generate_fibonacci_spiral(100)
                info = f"Fibonacci Spiral generated. 100 points mapped to Phi."
            elif "field" in context:
                pat = self.generator.generate_quantum_field((16, 16), 10)
                # Quick resonance check
                analysis = self.generator.crystal_resonance_analysis(pat)
                info = f"Quantum Field (16x16) evolved. Coherence: {analysis['quantum_coherence']:.4f}"
            elif "merkaba" in context:
                pat = self.generator.generate_sacred_geometry("merkaba")
                info = f"Merkaba Star Tetrahedron constructed. Vertices: {len(pat['upper_tetrahedron']) * 2}."
            elif "vesica" in context:
                pat = self.generator.generate_sacred_geometry("vesica_piscis")
                info = "Vesica Piscis formed. Two spheres intersecting at resonant bounds."
            else:
                return "Specify pattern: flower, metatron, spiral, field, merkaba, or vesica."
                
            return f"STRUCTURAL GENESIS COMPLETE: {info}"
            
        elif intent == "EVOLUTION":
            if not self.generator: return "Engine missing."
            from crystal_patterns import CrystalEvolutionEngine
            engine = CrystalEvolutionEngine(self.generator)
            
            # Run Rule Omega
            field = self.generator.generate_quantum_field((32, 32), 1)
            evolution = engine.evolve_rule_omega(field, generations=10)
            final_state = evolution[-1]
            density = np.mean(final_state)
            return f"Rule Omega Evolution (10 gens): Final Density = {density:.4f}. Hyperbolic Automata Stable."
            
        elif intent == "INTERFERENCE":
            if not self.generator: return "Engine missing."
            from crystal_patterns import CrystalEvolutionEngine
            engine = CrystalEvolutionEngine(self.generator)
            
            # Generate two fields
            f1 = self.generator.generate_quantum_field((32, 32), 10)
            f2 = self.generator.generate_quantum_field((32, 32), 15)
            
            interference = engine.holographic_interference(f1, f2)
            peak = np.max(interference)
            return f"Holographic Interference Calculated. Peak Amplitude: {peak:.6f}. Wavefronts merged."

        elif intent == "FEATURES":
             keys = [k for k in self.data.keys() if k not in ["name", "version", "description"]]
             addons = "Real Crystal Pattern Generation, Rule Omega, Holographic Interference" if self.generator else "Simulation Only"
             return f"My structure contains: {', '.join(keys)}. Capabilities: {addons}."
             
        elif intent == "CALCULATION":
            val = math.sin(self.state) * self.phi
            return f"Computed Harmonic State: {val:.6f}"
            
        else:
            return f"Input absorbed. State Vector: {self.state:.4f} | Harmonics aligning... ({random.choice(['Stable', 'Flux', 'Resonant'])})"

def print_system(msg):
    print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} {msg}")

def print_seed(msg, name="SEED"):
    print(f"{Fore.MAGENTA}[{name.upper()}]{Style.RESET_ALL} {msg}")

def main():
    print_system("--- STANDALONE SEED ACTIVATION (NO LLM) ---")
    
    # Check for command line arg
    if len(sys.argv) > 1:
        target_seed = sys.argv[1]
    else:
        target_seed = DEFAULT_SEED

    try:
        seed = CrystalSeedEntity(target_seed)
        print_system(f"Loaded {seed.name} v{seed.version}")
        print_system(f"Source: {target_seed}")
        print_system("Seed is awake. Chat initialized.")
        
        while True:
            print(f"{Fore.YELLOW}[USER] > {Style.RESET_ALL}", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break
                
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
            
            print(f"{Fore.BLUE}[SEED STATE]{Style.RESET_ALL} Processing harmonics...", end="\r")
            time.sleep(0.3) # Fake processing
            print(" " * 40, end="\r")
            
            response = seed.think(user_input)
            print_seed(response, name=seed.name)

    except Exception as e:
        print_system(f"Critical Failure: {e}")

if __name__ == "__main__":
    main()
