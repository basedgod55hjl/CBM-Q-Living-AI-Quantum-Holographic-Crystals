#!/usr/bin/env python3
"""
7DMH-QA Manifold Benchmark Suite
Performance testing for Crystal Architecture.
"""

import os
import sys
import time
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CRYSTAL_DIR)

from crystal_patterns import CrystalPatternGenerator, CrystalEvolutionEngine

class ManifoldBenchmark:
    """Comprehensive benchmarking for 7DMH-QA operations"""
    
    def __init__(self):
        self.gen = CrystalPatternGenerator(complexity=512)
        self.engine = CrystalEvolutionEngine(self.gen)
        self.results = {}
    
    def benchmark_manifold_generation(self, dimensions=7, resolutions=[8, 16, 32, 64]):
        """Benchmark holographic manifold generation at various resolutions"""
        print("\n📊 MANIFOLD GENERATION BENCHMARK")
        print("-" * 50)
        
        for res in resolutions:
            start = time.perf_counter()
            manifold = self.gen.generate_holographic_manifold(dimensions, res)
            elapsed = time.perf_counter() - start
            
            size_mb = manifold.nbytes / 1024 / 1024
            throughput = size_mb / elapsed
            
            print(f"  {dimensions}D @ {res}³ | {elapsed:.3f}s | {size_mb:.2f}MB | {throughput:.1f} MB/s")
            self.results[f'manifold_{res}'] = elapsed
    
    def benchmark_projection(self, sizes=[1000, 10000, 100000, 1000000]):
        """Benchmark manifold-constrained projection"""
        print("\n📊 PROJECTION BENCHMARK (S² Stability)")
        print("-" * 50)
        
        for size in sizes:
            tensor = np.random.randn(size, 64).astype(np.float32)
            
            start = time.perf_counter()
            projected = self.gen.manifold_constrained_projection(tensor)
            elapsed = time.perf_counter() - start
            
            ops = size * 64 * 3  # Approximate FLOP count
            gflops = (ops / elapsed) / 1e9
            
            print(f"  {size:>8,} elements | {elapsed:.4f}s | {gflops:.2f} GFLOP/s")
            self.results[f'projection_{size}'] = elapsed
    
    def benchmark_evolution(self, field_size=64, generations=[10, 50, 100]):
        """Benchmark Rule Omega evolution"""
        print("\n📊 EVOLUTION BENCHMARK (Rule Ω)")
        print("-" * 50)
        
        initial = np.random.rand(field_size, field_size)
        
        for gens in generations:
            start = time.perf_counter()
            evolution = self.engine.evolve_rule_omega(initial, gens)
            elapsed = time.perf_counter() - start
            
            gens_per_sec = gens / elapsed
            print(f"  {field_size}x{field_size} × {gens} gens | {elapsed:.3f}s | {gens_per_sec:.1f} gen/s")
            self.results[f'evolution_{gens}'] = elapsed
    
    def benchmark_interference(self, sizes=[32, 64, 128, 256]):
        """Benchmark holographic interference"""
        print("\n📊 HOLOGRAPHIC INTERFERENCE BENCHMARK")
        print("-" * 50)
        
        for size in sizes:
            p1 = np.random.rand(size, size)
            p2 = np.random.rand(size, size)
            
            start = time.perf_counter()
            interference = self.engine.holographic_interference(p1, p2)
            elapsed = time.perf_counter() - start
            
            pixels = size * size
            throughput = pixels / elapsed / 1e6
            
            print(f"  {size}x{size} | {elapsed:.4f}s | {throughput:.2f} Mpixels/s")
            self.results[f'interference_{size}'] = elapsed
    
    def benchmark_resonance_analysis(self, sizes=[32, 64, 128]):
        """Benchmark crystal resonance analysis"""
        print("\n📊 RESONANCE ANALYSIS BENCHMARK")
        print("-" * 50)
        
        for size in sizes:
            pattern = np.random.rand(size, size)
            
            start = time.perf_counter()
            analysis = self.gen.crystal_resonance_analysis(pattern)
            elapsed = time.perf_counter() - start
            
            print(f"  {size}x{size} | {elapsed:.4f}s | Φ={analysis['phi_resonance']:.2f}")
            self.results[f'resonance_{size}'] = elapsed
    
    def run_all(self):
        """Run complete benchmark suite"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║       7DMH-QA MANIFOLD BENCHMARK SUITE                       ║
║   Manifold-Constrained Holographic Quantum Architecture      ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        self.benchmark_manifold_generation()
        self.benchmark_projection()
        self.benchmark_evolution()
        self.benchmark_interference()
        self.benchmark_resonance_analysis()
        
        self.summary()
    
    def summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("  BENCHMARK SUMMARY")
        print("="*60)
        
        total = sum(self.results.values())
        print(f"  Total Runtime: {total:.2f}s")
        print(f"  Tests Completed: {len(self.results)}")
        
        # Find bottleneck
        slowest = max(self.results, key=self.results.get)
        print(f"  Slowest: {slowest} ({self.results[slowest]:.3f}s)")
        
        print("="*60)
        print("\n✅ Benchmark Complete")

def main():
    bench = ManifoldBenchmark()
    bench.run_all()

if __name__ == "__main__":
    main()
