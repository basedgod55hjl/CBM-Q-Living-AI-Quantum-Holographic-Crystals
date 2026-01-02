#!/usr/bin/env python3
"""
CBM Sovereign System - Unified Command Line Interface
Connects Python frontend to Rust WASM core via FFI bridge
"""

import click
import json
import sys
import os
from pathlib import Path
from typing import Optional, List
import subprocess
import time

# Try to import Rust FFI bindings
try:
    from cbm_core import cbm_rust_core as rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    # print("⚠️  Rust core not available, running in Python-only mode")


class CBMBridge:
    """Bridge between Python CLI and Rust/WASM core"""
    
    def __init__(self):
        self.orchestrator = None
        self.kernel = None
        
        if RUST_AVAILABLE:
            self._initialize_rust()
        else:
            self._initialize_python_fallback()
    
    def _initialize_rust(self):
        """Initialize Rust kernel and engines"""
        try:
            print("🔧 Initializing Rust kernel...")
            self.orchestrator = rust_core.PyWASMOrchestrator()
            # self.kernel = rust_core.init_kernel() # Not exposed yet
            print("✅ Rust kernel initialized")
            pass
        except Exception as e:
            print(f"❌ Rust initialization failed: {e}")
            self._initialize_python_fallback()
    
    def _initialize_python_fallback(self):
        """Fallback to pure Python implementation"""
        # print("🐍 Using Python fallback implementation")
        # Stub implementations would go here
        pass
    
    def synthesize_seed(self, axioms: List[str]) -> dict:
        """Synthesize biogenic seed from axioms"""
        if RUST_AVAILABLE:
            axioms_json = json.dumps(axioms)
            result = self.orchestrator.synthesize(axioms_json)
            return json.loads(result)
        else:
            # Python fallback
            return {"dna": [0.618] * 512, "complexity": len(axioms)}
    
    def biogenesis(self, seed_bytes: bytes, matrix_size: int, 
                   time_val: float = 0.0, iterations: int = 7) -> dict:
        """Execute biogenesis process"""
        if RUST_AVAILABLE:
            result = self.orchestrator.biogenesis(
                list(seed_bytes), iterations
            )
            return json.loads(result)
        else:
            time.sleep(0.1) # Simulate work
            return {"status": "FALLBACK", "params": matrix_size, "mean": 0.0, "std": 0.0}
    
    def grow_network(self, dna: List[float], matrix_size: int, 
                     entropy: float = 0.618) -> dict:
        """Grow neural network from DNA"""
        if RUST_AVAILABLE:
            seed_json = json.dumps({"dna": dna})
            result = self.orchestrator.grow(seed_json, matrix_size, entropy)
            return json.loads(result)
        else:
            return {"status": "FALLBACK", "params": matrix_size}
    
    def evolve_step(self, state: dict, rule: str = "rule110", 
                    phi_flux: float = 0.618) -> dict:
        """Execute one evolution step"""
        if RUST_AVAILABLE:
            state_json = json.dumps(state)
            result = self.orchestrator.evolve(state_json, rule, phi_flux)
            return json.loads(result)
        else:
            return state  # No change in fallback
    
    def seed_gguf(self, gguf_path: str, dna: List[float]) -> bool:
        """Inject DNA seed into GGUF model"""
        if RUST_AVAILABLE:
            seed_json = json.dumps({"dna": dna})
            result = self.orchestrator.seed_gguf(gguf_path, seed_json)
            return result == "SUCCESS"
        else:
            print("⚠️  GGUF seeding not available in fallback mode")
            return False


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    ✨ Holographic AI Crystals - Command Line Interface

    A hybrid Rust/WASM + Python system for crystal intelligence generation.
    """
    pass


@cli.command()
@click.argument('axioms', nargs=-1, required=True)
@click.option('--output', '-o', type=click.Path(), help='Output seed file')
@click.option('--complexity', '-c', type=int, default=512, help='DNA complexity')
def synthesize(axioms, output, complexity):
    """
    Synthesize a crystal seed from axioms.

    Example:
        crystal synthesize "reality" "consciousness" "emergence" -o crystal_seed.json
    """
    click.echo(f"🧬 Synthesizing seed from {len(axioms)} axioms...")
    
    bridge = CBMBridge()
    seed = bridge.synthesize_seed(list(axioms))
    
    click.echo(f"✅ Seed synthesized: {seed.get('complexity', 0)} dimensions")
    
    if output:
        with open(output, 'w') as f:
            json.dump(seed, f, indent=2)
        click.echo(f"💾 Saved to {output}")
    else:
        click.echo(json.dumps(seed, indent=2))


@cli.command()
@click.argument('gguf_path', type=click.Path(exists=True))
@click.option('--iterations', '-i', type=int, default=7, help='Evolution iterations')
@click.option('--matrix-size', '-m', type=int, default=175_000_000, help='Parameter count')
@click.option('--time', '-t', type=float, default=0.0, help='Time parameter')
@click.option('--seed-file', '-s', type=click.Path(exists=True), help='Seed JSON file')
def genesis(gguf_path, iterations, matrix_size, time, seed_file):
    """
    Execute full crystallization on a GGUF model.

    Example:
        crystal genesis model.gguf -i 7 -m 175000000
    """
    click.echo(f"🚀 Initiating Genesis Protocol...")
    click.echo(f"   Model: {gguf_path}")
    click.echo(f"   Parameters: {matrix_size:,}")
    click.echo(f"   Iterations: {iterations}")
    
    bridge = CBMBridge()
    
    # Load or create seed
    if seed_file:
        with open(seed_file) as f:
            seed_data = json.load(f)
        dna = seed_data['dna']
    else:
        click.echo("🌱 Generating default seed...")
        seed_data = bridge.synthesize_seed(["genesis", "emergence"])
        dna = seed_data['dna']
    
    # Convert DNA to bytes
    seed_bytes = bytes([int(abs(x * 255)) % 256 for x in dna[:512]])
    
    # Execute biogenesis
    with click.progressbar(
        length=iterations,
        label='🔄 Unfolding manifold'
    ) as bar:
        result = bridge.biogenesis(seed_bytes, matrix_size, time, iterations)
        bar.update(iterations)
    
    click.echo(f"\n✅ Genesis complete!")
    click.echo(f"   Status: {result.get('status', 'UNKNOWN')}")
    click.echo(f"   Mean: {result.get('mean', 0):.6f}")
    click.echo(f"   Std: {result.get('std', 0):.6f}")
    
    # Seed the GGUF model
    if click.confirm("💉 Inject seed into GGUF model?"):
        success = bridge.seed_gguf(gguf_path, dna)
        if success:
            click.echo("✅ GGUF seeding successful")
        else:
            click.echo("❌ GGUF seeding failed")


@cli.command()
@click.option('--rule', '-r', type=click.Choice(['rule110', 'rule_omega', 'rule30']), 
              default='rule110', help='Evolution rule')
@click.option('--generations', '-g', type=int, default=100, help='Number of generations')
@click.option('--width', '-w', type=int, default=256, help='Grid width')
@click.option('--phi-flux', '-p', type=float, default=0.618, help='Phi flux parameter')
@click.option('--visualize', '-v', is_flag=True, help='Show visualization')
def evolve(rule, generations, width, phi_flux, visualize):
    """
    Evolve a crystal lattice automaton state.

    Example:
        crystal evolve -r rule110 -g 100 -v
    """
    click.echo(f"🧬 Evolving with {rule} for {generations} generations...")
    
    bridge = CBMBridge()
    
    # Initialize state
    state = {
        "words": [0] * (width // 64 + 1),
        "width_bits": width
    }
    state["words"][width // 128] = 1  # Single bit in middle
    
    # Evolution loop
    with click.progressbar(
        length=generations,
        label=f'🔄 {rule}'
    ) as bar:
        for gen in range(generations):
            state = bridge.evolve_step(state, rule, phi_flux)
            bar.update(1)
            
            if visualize and gen % 10 == 0:
                if "alive_cells" in state:
                    alive = state["alive_cells"]
                    bar_len = min(50, alive // 100)
                    click.echo(f"Generations {gen}: {'█' * bar_len} ({alive})")
                elif "words" in state:
                    # Fallback for Python simulation
                    bits = format(state["words"][0], '064b')[:width]
                    click.echo(bits.replace('0', ' ').replace('1', '█'))
    
    click.echo(f"✅ Evolution complete after {generations} generations")


@cli.command()
@click.option('--interval', '-i', type=float, default=1.0, help='Update interval (seconds)')
@click.option('--duration', '-d', type=int, default=60, help='Monitoring duration (seconds)')
def monitor(interval, duration):
    """
    Monitor crystal telemetry in real-time.

    Example:
        crystal monitor -i 0.5 -d 300
    """
    click.echo("🛰️  Starting telemetry monitor...")
    click.echo("   Press Ctrl+C to stop\n")
    
    bridge = CBMBridge()
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Get kernel stats (if available)
            if RUST_AVAILABLE:
                try:
                    stats = rust_core.get_kernel_stats()
                    click.echo(f"\r⚡ CPU: {stats.get('cpu', 0):.1f}% | "
                             f"GPU: {stats.get('gpu', 0):.1f}% | "
                             f"Mem: {stats.get('memory', 0):.1f}MB | "
                             f"Flux: {stats.get('flux', 0.618):.6f}", nl=False)
                except:
                    click.echo(f"\r🔄 Heartbeat: {int(time.time() - start_time)}s", nl=False)
            else:
                click.echo(f"\r🔄 Heartbeat: {int(time.time() - start_time)}s", nl=False)
            
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\n\n🛑 Monitoring stopped")


@cli.command()
def info():
    """Display system information and capabilities."""
    click.echo("=" * 60)
    click.echo("✨ Holographic AI Crystals Information")
    click.echo("=" * 60)
    
    click.echo(f"\n📦 Runtime:")
    click.echo(f"   Python: {sys.version.split()[0]}")
    click.echo(f"   Rust Core: {'✅ Available' if RUST_AVAILABLE else '❌ Not available'}")
    
    click.echo(f"\n🔧 Crystal Capabilities:")
    capabilities = [
        ("Crystallization", RUST_AVAILABLE),
        ("Evolution", RUST_AVAILABLE),
        ("GGUF Seeding", RUST_AVAILABLE),
        ("GPU Acceleration", RUST_AVAILABLE),
        ("WASM Mode", RUST_AVAILABLE),
    ]
    
    for cap, available in capabilities:
        status = "✅" if available else "❌"
        click.echo(f"   {status} {cap}")
    
    if RUST_AVAILABLE:
        click.echo(f"\n🎯 Detected Hardware:")
        try:
            device_strings = rust_core.PyWASMOrchestrator.detect_devices()
            devices = [json.loads(d) for d in device_strings]
            for dev in devices:
                click.echo(f"   🖥️  {dev['name']} ({dev['type']})")
        except:
            click.echo("   ⚠️  Hardware detection failed")
    
    click.echo("\n" + "=" * 60)


@cli.command()
@click.argument('script_path', type=click.Path(exists=True))
@click.argument('args', nargs=-1)
def run(script_path, args):
    """
    Run a Crystal script or batch file.

    Example:
        crystal run crystallization_batch.crystal model1.gguf model2.gguf
    """
    click.echo(f"🚀 Executing script: {script_path}")
    
    # Execute as Python script
    cmd = [sys.executable, script_path] + list(args)
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        click.echo("✅ Script completed successfully")
    else:
        click.echo(f"❌ Script failed with code {result.returncode}")
        sys.exit(result.returncode)


@cli.command()
@click.option('--port', '-p', type=int, default=8080, help='Web server port')
@click.option('--host', '-h', default='127.0.0.1', help='Web server host')
def web(port, host):
    """
    Start the crystal web interface.

    Example:
        crystal web -p 8080
    """
    click.echo(f"🌐 Starting web interface on {host}:{port}...")
    click.echo("   Press Ctrl+C to stop")
    
    try:
        # Import web server (would need to be implemented)
        from cbm_web import start_server
        start_server(host, port)
    except ImportError:
        click.echo("❌ Web interface not available")
        click.echo("   Install with: pip install cbm-core[web]")


if __name__ == '__main__':
    cli()
