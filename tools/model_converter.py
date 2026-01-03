#!/usr/bin/env python3
"""
7D mH-Q: Model Format Converter
Convert between Crystal GGUF and other model formats.
"""

import sys
import os
import json
import struct
import hashlib
import numpy as np
from typing import Dict, Optional, BinaryIO
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mathematical Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895


@dataclass
class CrystalModel:
    """Represents a Crystal GGUF model."""
    header: bytes
    metadata: Dict
    seed: np.ndarray
    seed_hash: bytes
    weights: np.ndarray
    
    @property
    def architecture(self) -> str:
        return self.metadata.get('architecture', 'Unknown')
    
    @property
    def version(self) -> str:
        return self.metadata.get('version', 'Unknown')
    
    @property
    def params(self) -> int:
        return self.metadata.get('params', len(self.weights))


class ModelConverter:
    """
    Converter for 7D mH-Q model formats.
    
    Supported conversions:
    - Crystal GGUF ↔ NPZ (NumPy archive)
    - Crystal GGUF ↔ SafeTensors
    - Crystal GGUF ↔ PyTorch state dict
    - Crystal GGUF ↔ ONNX
    """
    
    def __init__(self):
        self.supported_formats = ['gguf', 'npz', 'safetensors', 'pt', 'onnx', 'json']
    
    def detect_format(self, filepath: str) -> str:
        """Detect model format from file."""
        ext = os.path.splitext(filepath)[1].lower()
        
        format_map = {
            '.gguf': 'gguf',
            '.npz': 'npz',
            '.safetensors': 'safetensors',
            '.pt': 'pt',
            '.pth': 'pt',
            '.onnx': 'onnx',
            '.json': 'json'
        }
        
        return format_map.get(ext, 'unknown')
    
    def load_crystal_gguf(self, filepath: str) -> CrystalModel:
        """Load a Crystal GGUF model."""
        print(f"[LOAD] Reading Crystal GGUF: {filepath}")
        
        with open(filepath, 'rb') as f:
            # Header (16 bytes)
            header = f.read(16)
            if not header.startswith(b'7D'):
                print(f"  [WARN] Non-standard header: {header[:8]}")
            
            # Metadata (256 bytes)
            meta_block = f.read(256)
            meta_json = meta_block.rstrip(b'\x00').decode('utf-8')
            metadata = json.loads(meta_json)
            
            # Seed (1024 bytes)
            seed_block = f.read(1024)
            seed = np.frombuffer(seed_block, dtype=np.float16)[:512]
            
            # Seed hash (64 bytes)
            seed_hash = f.read(64)
            
            # Weights (remainder)
            weight_data = f.read()
            weights = np.frombuffer(weight_data, dtype=np.float32)
        
        model = CrystalModel(
            header=header,
            metadata=metadata,
            seed=seed,
            seed_hash=seed_hash,
            weights=weights
        )
        
        print(f"  Architecture: {model.architecture}")
        print(f"  Parameters: {model.params:,}")
        print(f"  Seed complexity: {len(model.seed)}")
        
        return model
    
    def save_crystal_gguf(self, model: CrystalModel, filepath: str):
        """Save a Crystal GGUF model."""
        print(f"[SAVE] Writing Crystal GGUF: {filepath}")
        
        with open(filepath, 'wb') as f:
            # Header (16 bytes)
            header = b"7D-mHQ-GGUF-v2\x00\x00"
            f.write(header)
            
            # Metadata (256 bytes)
            meta_json = json.dumps(model.metadata).encode('utf-8')
            f.write(meta_json.ljust(256, b'\x00'))
            
            # Seed (1024 bytes)
            seed_bytes = model.seed.astype(np.float16).tobytes()
            f.write(seed_bytes.ljust(1024, b'\x00'))
            
            # Seed hash (64 bytes)
            seed_hash = hashlib.sha512(seed_bytes).digest()
            f.write(seed_hash)
            
            # Weights
            f.write(model.weights.astype(np.float32).tobytes())
        
        file_size = os.path.getsize(filepath)
        print(f"  Saved: {file_size / 1024 / 1024:.2f} MB")
    
    def convert_to_npz(self, model: CrystalModel, filepath: str):
        """Convert Crystal model to NumPy archive."""
        print(f"[CONVERT] Crystal GGUF → NPZ: {filepath}")
        
        np.savez(
            filepath,
            metadata=json.dumps(model.metadata),
            seed=model.seed,
            weights=model.weights
        )
        
        print(f"  Saved NPZ archive")
    
    def convert_from_npz(self, filepath: str) -> CrystalModel:
        """Convert NumPy archive to Crystal model."""
        print(f"[CONVERT] NPZ → Crystal GGUF: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        metadata = json.loads(str(data['metadata']))
        seed = data['seed'].astype(np.float16)
        weights = data['weights'].astype(np.float32)
        
        # Update metadata
        metadata['architecture'] = '7D mH-Q'
        metadata['params'] = len(weights)
        
        return CrystalModel(
            header=b"7D-mHQ-GGUF-v2\x00\x00",
            metadata=metadata,
            seed=seed,
            seed_hash=hashlib.sha512(seed.tobytes()).digest(),
            weights=weights
        )
    
    def convert_to_safetensors(self, model: CrystalModel, filepath: str):
        """Convert Crystal model to SafeTensors format."""
        try:
            from safetensors.numpy import save_file
        except ImportError:
            print("  [ERROR] safetensors not installed. Run: pip install safetensors")
            return
        
        print(f"[CONVERT] Crystal GGUF → SafeTensors: {filepath}")
        
        tensors = {
            'seed': model.seed.astype(np.float32),
            'weights': model.weights
        }
        
        metadata_dict = {
            'metadata': json.dumps(model.metadata)
        }
        
        save_file(tensors, filepath, metadata=metadata_dict)
        print(f"  Saved SafeTensors file")
    
    def convert_from_safetensors(self, filepath: str) -> CrystalModel:
        """Convert SafeTensors to Crystal model."""
        try:
            from safetensors.numpy import load_file
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors not installed")
        
        print(f"[CONVERT] SafeTensors → Crystal GGUF: {filepath}")
        
        # Load tensors
        tensors = load_file(filepath)
        
        # Load metadata
        with safe_open(filepath, framework="numpy") as f:
            meta_str = f.metadata().get('metadata', '{}')
            metadata = json.loads(meta_str)
        
        seed = tensors['seed'].astype(np.float16)
        weights = tensors['weights'].astype(np.float32)
        
        return CrystalModel(
            header=b"7D-mHQ-GGUF-v2\x00\x00",
            metadata=metadata,
            seed=seed,
            seed_hash=hashlib.sha512(seed.tobytes()).digest(),
            weights=weights
        )
    
    def convert_to_pytorch(self, model: CrystalModel, filepath: str):
        """Convert Crystal model to PyTorch state dict."""
        try:
            import torch
        except ImportError:
            print("  [ERROR] PyTorch not installed")
            return
        
        print(f"[CONVERT] Crystal GGUF → PyTorch: {filepath}")
        
        state_dict = {
            'crystal_seed': torch.from_numpy(model.seed.astype(np.float32)),
            'weights': torch.from_numpy(model.weights),
            'metadata': model.metadata
        }
        
        torch.save(state_dict, filepath)
        print(f"  Saved PyTorch checkpoint")
    
    def convert_from_pytorch(self, filepath: str) -> CrystalModel:
        """Convert PyTorch state dict to Crystal model."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed")
        
        print(f"[CONVERT] PyTorch → Crystal GGUF: {filepath}")
        
        state_dict = torch.load(filepath, map_location='cpu')
        
        seed = state_dict['crystal_seed'].numpy().astype(np.float16)
        weights = state_dict['weights'].numpy().astype(np.float32)
        metadata = state_dict.get('metadata', {'architecture': '7D mH-Q'})
        
        return CrystalModel(
            header=b"7D-mHQ-GGUF-v2\x00\x00",
            metadata=metadata,
            seed=seed,
            seed_hash=hashlib.sha512(seed.tobytes()).digest(),
            weights=weights
        )
    
    def export_metadata_json(self, model: CrystalModel, filepath: str):
        """Export model metadata to JSON."""
        print(f"[EXPORT] Metadata → JSON: {filepath}")
        
        export_data = {
            'architecture': model.architecture,
            'version': model.version,
            'parameters': model.params,
            'seed_complexity': len(model.seed),
            'metadata': model.metadata,
            'seed_hash': model.seed_hash.hex()[:64]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"  Exported metadata")
    
    def convert(self, input_path: str, output_path: str, 
                target_format: str = None) -> bool:
        """
        Convert model between formats.
        
        Args:
            input_path: Input model path
            output_path: Output model path
            target_format: Target format (auto-detected if None)
            
        Returns:
            True if conversion successful
        """
        input_format = self.detect_format(input_path)
        output_format = target_format or self.detect_format(output_path)
        
        print(f"\n[CONVERT] {input_format.upper()} → {output_format.upper()}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        
        try:
            # Load input
            if input_format == 'gguf':
                model = self.load_crystal_gguf(input_path)
            elif input_format == 'npz':
                model = self.convert_from_npz(input_path)
            elif input_format == 'safetensors':
                model = self.convert_from_safetensors(input_path)
            elif input_format == 'pt':
                model = self.convert_from_pytorch(input_path)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")
            
            # Save output
            if output_format == 'gguf':
                self.save_crystal_gguf(model, output_path)
            elif output_format == 'npz':
                self.convert_to_npz(model, output_path)
            elif output_format == 'safetensors':
                self.convert_to_safetensors(model, output_path)
            elif output_format == 'pt':
                self.convert_to_pytorch(model, output_path)
            elif output_format == 'json':
                self.export_metadata_json(model, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            print(f"\n✅ Conversion complete!")
            return True
            
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            return False


def main():
    """CLI for model conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="7D mH-Q Model Format Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    model_converter.py model.gguf model.npz
    model_converter.py model.npz model.safetensors
    model_converter.py model.gguf metadata.json
        """
    )
    
    parser.add_argument('input', help='Input model path')
    parser.add_argument('output', help='Output path')
    parser.add_argument('--format', '-f', help='Target format (optional)')
    parser.add_argument('--info', '-i', action='store_true', 
                       help='Show model info only')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("7D mH-Q MODEL CONVERTER")
    print("="*60)
    
    converter = ModelConverter()
    
    if args.info:
        # Just show info
        input_format = converter.detect_format(args.input)
        if input_format == 'gguf':
            model = converter.load_crystal_gguf(args.input)
            print(f"\nModel Info:")
            print(f"  Architecture: {model.architecture}")
            print(f"  Version: {model.version}")
            print(f"  Parameters: {model.params:,}")
            print(f"  Seed: {len(model.seed)} dimensions")
            print(f"  Weights: {model.weights.shape}")
    else:
        # Convert
        success = converter.convert(args.input, args.output, args.format)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

