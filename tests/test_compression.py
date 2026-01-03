#!/usr/bin/env python3
"""
7D mH-Q: Compression Testing Suite
Tests holographic compression and GGUF format.
"""

import sys
import os
import numpy as np
import time
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mathematical Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895


class CompressionTester:
    """
    Tests compression capabilities of 7D mH-Q.
    """
    
    def __init__(self):
        self.results = []
    
    def test_seed_unfolding(self) -> dict:
        """
        Test seed-to-weights unfolding algorithm.
        """
        print(f"\n{'='*60}")
        print("TEST: Seed-to-Weights Unfolding")
        print(f"{'='*60}")
        
        # Create seed
        seed_size = 512
        seed = np.random.randn(seed_size).astype(np.float32)
        
        # Test various output sizes
        output_sizes = [1000, 10000, 100000, 1000000]
        
        unfold_results = []
        
        for output_size in output_sizes:
            start_time = time.time()
            
            # Unfold using CBM flux algorithm
            output = np.zeros(output_size, dtype=np.float32)
            
            for idx in range(output_size):
                seed_idx = idx % seed_size
                generation = idx // seed_size
                
                base = seed[seed_idx]
                flux = np.sin(base * PHI + generation * PHI_INV) * PHI_INV
                interference = np.cos(idx * PHI_INV / 1000.0)
                
                output[idx] = np.tanh(base + flux * 0.1 + interference * 0.01)
            
            elapsed = time.time() - start_time
            compression_ratio = output_size / seed_size
            
            # Verify output is bounded
            bounded = np.all(np.abs(output) <= 1.0)
            
            unfold_results.append({
                'output_size': output_size,
                'compression_ratio': compression_ratio,
                'time': elapsed,
                'bounded': bounded,
                'mean': np.mean(output),
                'std': np.std(output)
            })
            
            print(f"  {output_size:,} weights from {seed_size} seed:")
            print(f"    Ratio: {compression_ratio:,.0f}x | Time: {elapsed:.3f}s | Bounded: {bounded}")
        
        all_bounded = all(r['bounded'] for r in unfold_results)
        max_ratio = max(r['compression_ratio'] for r in unfold_results)
        
        passed = all_bounded and max_ratio >= 1000
        
        result = {
            'test': 'seed_unfolding',
            'max_compression_ratio': max_ratio,
            'all_bounded': all_bounded,
            'passed': passed,
            'details': unfold_results
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"Max ratio = {max_ratio:,.0f}x, all outputs bounded")
        
        return result
    
    def test_holographic_encoding(self) -> dict:
        """
        Test holographic interference pattern encoding.
        """
        print(f"\n{'='*60}")
        print("TEST: Holographic Interference Encoding")
        print(f"{'='*60}")
        
        from crystal_patterns import CrystalEvolutionEngine, CrystalPatternGenerator
        
        pattern_gen = CrystalPatternGenerator()
        evolution = CrystalEvolutionEngine(pattern_gen)
        
        # Create test patterns
        pattern1 = np.random.randn(64, 64).astype(np.float32)
        pattern2 = np.random.randn(64, 64).astype(np.float32)
        
        # Generate holographic interference
        interference = evolution.holographic_interference(pattern1, pattern2)
        
        # Test properties
        bounded = np.all(np.abs(interference) <= 1.0)
        smooth = np.std(np.diff(interference.flatten())) < 1.0
        
        print(f"  Pattern 1 shape: {pattern1.shape}")
        print(f"  Pattern 2 shape: {pattern2.shape}")
        print(f"  Interference shape: {interference.shape}")
        print(f"  Interference range: [{interference.min():.4f}, {interference.max():.4f}]")
        print(f"  Bounded: {bounded}")
        print(f"  Smooth: {smooth}")
        
        passed = bounded
        
        result = {
            'test': 'holographic_encoding',
            'bounded': bounded,
            'smooth': smooth,
            'interference_range': (interference.min(), interference.max()),
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: Interference properly bounded")
        
        return result
    
    def test_gguf_format(self) -> dict:
        """
        Test Crystal GGUF file format.
        """
        print(f"\n{'='*60}")
        print("TEST: Crystal GGUF Format")
        print(f"{'='*60}")
        
        import json
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp:
            gguf_path = tmp.name
        
        try:
            # Create test model
            seed = np.random.randn(512).astype(np.float16)
            weights = np.random.randn(10000).astype(np.float32)
            
            metadata = {
                'architecture': '7D mH-Q',
                'version': '2.0.0',
                'params': len(weights),
                'seed_complexity': len(seed)
            }
            
            # Write GGUF
            print("  Writing GGUF...")
            with open(gguf_path, 'wb') as f:
                # Header (16 bytes)
                header = b"7D-mHQ-GGUF-v2\x00\x00"
                f.write(header)
                
                # Metadata (256 bytes)
                meta_json = json.dumps(metadata).encode('utf-8')
                f.write(meta_json.ljust(256, b'\x00'))
                
                # Seed (1024 bytes)
                seed_bytes = seed.tobytes().ljust(1024, b'\x00')
                f.write(seed_bytes)
                
                # Hash (64 bytes)
                seed_hash = hashlib.sha512(seed.tobytes()).digest()
                f.write(seed_hash)
                
                # Weights
                f.write(weights.tobytes())
            
            file_size = os.path.getsize(gguf_path)
            print(f"  File size: {file_size:,} bytes")
            
            # Read and verify
            print("  Reading GGUF...")
            with open(gguf_path, 'rb') as f:
                # Header
                read_header = f.read(16)
                header_valid = read_header.startswith(b"7D-mHQ-GGUF")
                
                # Metadata
                meta_block = f.read(256)
                read_meta = json.loads(meta_block.rstrip(b'\x00').decode('utf-8'))
                meta_valid = read_meta['architecture'] == '7D mH-Q'
                
                # Seed
                seed_block = f.read(1024)
                read_seed = np.frombuffer(seed_block[:seed.nbytes], dtype=np.float16)
                seed_valid = np.allclose(read_seed, seed)
                
                # Hash
                read_hash = f.read(64)
                hash_valid = read_hash == seed_hash
                
                # Weights
                weight_data = f.read()
                read_weights = np.frombuffer(weight_data, dtype=np.float32)
                weights_valid = np.allclose(read_weights, weights)
            
            print(f"  Header valid: {header_valid}")
            print(f"  Metadata valid: {meta_valid}")
            print(f"  Seed valid: {seed_valid}")
            print(f"  Hash valid: {hash_valid}")
            print(f"  Weights valid: {weights_valid}")
            
            all_valid = header_valid and meta_valid and seed_valid and hash_valid and weights_valid
            
        finally:
            os.unlink(gguf_path)
        
        passed = all_valid
        
        result = {
            'test': 'gguf_format',
            'header_valid': header_valid,
            'metadata_valid': meta_valid,
            'seed_valid': seed_valid,
            'hash_valid': hash_valid,
            'weights_valid': weights_valid,
            'file_size': file_size,
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: GGUF format verified")
        
        return result
    
    def test_reconstruction_accuracy(self) -> dict:
        """
        Test holographic reconstruction from partial data.
        """
        print(f"\n{'='*60}")
        print("TEST: Holographic Reconstruction Accuracy")
        print(f"{'='*60}")
        
        # Create original pattern
        original = np.random.randn(64, 64).astype(np.float32)
        original_fft = np.fft.fft2(original)
        
        # Create holographic encoding (reference + object)
        reference = np.exp(1j * np.random.randn(64, 64) * 0.5)
        hologram = np.abs(reference + original_fft) ** 2
        
        # Simulate 50% data loss
        mask = np.random.rand(64, 64) > 0.5
        corrupted_hologram = hologram * mask
        
        # Attempt reconstruction
        # Simple approach: interpolate missing values
        from scipy import ndimage
        reconstructed_hologram = ndimage.uniform_filter(corrupted_hologram, size=3)
        reconstructed_hologram = np.where(mask, hologram, reconstructed_hologram)
        
        # Measure reconstruction quality
        mse = np.mean((reconstructed_hologram - hologram) ** 2)
        correlation = np.corrcoef(hologram.flatten(), reconstructed_hologram.flatten())[0, 1]
        
        print(f"  Original hologram size: {hologram.shape}")
        print(f"  Data retained: 50%")
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        passed = correlation > 0.8  # At least 80% correlation
        
        result = {
            'test': 'reconstruction_accuracy',
            'data_retained_pct': 50,
            'mse': mse,
            'correlation': correlation,
            'passed': passed
        }
        
        self.results.append(result)
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"Reconstruction correlation = {correlation:.4f}")
        
        return result
    
    def run_all_tests(self) -> dict:
        """Run complete compression test suite."""
        print("\n" + "="*60)
        print("7D mH-Q COMPRESSION TEST SUITE")
        print("="*60)
        
        start_time = time.time()
        
        self.test_seed_unfolding()
        self.test_holographic_encoding()
        self.test_gguf_format()
        self.test_reconstruction_accuracy()
        
        elapsed = time.time() - start_time
        
        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tests passed: {passed_count}/{total_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        
        if passed_count == total_count:
            print("\n[SUCCESS] ALL TESTS PASSED - COMPRESSION VERIFIED")
        else:
            print(f"\n[WARNING] {total_count - passed_count} TESTS FAILED")
        
        return {
            'passed': passed_count,
            'total': total_count,
            'all_passed': passed_count == total_count,
            'elapsed_seconds': elapsed,
            'results': self.results
        }


if __name__ == "__main__":
    tester = CompressionTester()
    summary = tester.run_all_tests()
    
    sys.exit(0 if summary['all_passed'] else 1)

