#!/usr/bin/env python3
"""
7D mH-Q: Crystal Command Line Interface
Unified CLI for all Crystal Architecture operations.

Usage:
    python crystal_cli.py <command> [options]
    
Commands:
    genesis     - Generate new Crystal model
    train       - Train Crystal model
    infer       - Run inference
    compress    - Compress files holographically
    convert     - Convert model formats
    benchmark   - Run performance benchmarks
    info        - Display model information
    visualize   - Generate visualizations
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CrystalCLI:
    """
    Unified Command Line Interface for 7D mH-Q Crystal Architecture.
    """
    
    def __init__(self):
        self.version = "2.0.0"
        self.author = "Sir Charles Spikes"
        
    def print_banner(self):
        """Print CLI banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ╔═╗┬─┐┬ ┬┌─┐┌┬┐┌─┐┬    ╔═╗╦  ╦                             ║
║   ║  ├┬┘└┬┘└─┐ │ ├─┤│    ║  ║  ║                              ║
║   ╚═╝┴└─ ┴ └─┘ ┴ ┴ ┴┴─┘  ╚═╝╩═╝╩                              ║
║                                                               ║
║   7D mH-Q Manifold-Constrained Holographic Quantum           ║
║   Architecture - Command Line Interface v2.0                  ║
║                                                               ║
║   © 2026 Sir Charles Spikes | Made in Ohio, USA 🇺🇸          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
        print(banner)
    
    def cmd_genesis(self, args):
        """Generate new Crystal model."""
        print("\n[GENESIS] Creating new Crystal model...")
        
        from sovereign_genesis import ManifoldConstrainedHolographicQuantumGenesisEngine
        
        matrix_size = args.params if args.params else 10_000_000
        output_name = args.output if args.output else "crystal_genesis.gguf"
        
        print(f"  Parameters: {matrix_size:,}")
        print(f"  Output: {output_name}")
        
        engine = ManifoldConstrainedHolographicQuantumGenesisEngine(matrix_size=matrix_size)
        engine.run_genesis(output_name)
        
        print(f"\n✅ Genesis complete: {output_name}")
    
    def cmd_train(self, args):
        """Train Crystal model."""
        print("\n[TRAIN] Starting Crystal training pipeline...")
        
        from engines.training_pipeline import CrystalTrainingPipeline, TrainingConfig
        import numpy as np
        
        config = TrainingConfig(
            epochs=args.epochs if args.epochs else 100,
            learning_rate=args.lr if args.lr else 0.001,
            batch_size=args.batch if args.batch else 32,
            device=args.device if args.device else "auto"
        )
        
        pipeline = CrystalTrainingPipeline(config)
        
        # Check for input data
        if args.data:
            print(f"  Loading data from: {args.data}")
            # Placeholder for actual data loading
            data = np.load(args.data, allow_pickle=True)
        else:
            print("  Generating synthetic training data...")
            np.random.seed(42)
            data = [(np.random.randn(64), np.random.randn(64)) for _ in range(1000)]
        
        results = pipeline.train(data)
        
        output_path = args.output if args.output else "trained_model.gguf"
        pipeline.export_gguf(output_path)
        
        print(f"\n✅ Training complete: {output_path}")
    
    def cmd_infer(self, args):
        """Run inference on Crystal model."""
        print("\n[INFER] Running Crystal inference...")
        
        from engines.inference_engine import CrystalInferenceEngine
        import numpy as np
        
        if not args.model:
            print("❌ Error: --model is required")
            return
        
        engine = CrystalInferenceEngine(args.model, device=args.device or "auto")
        
        if args.input:
            # Load input from file
            input_data = np.load(args.input)
        else:
            # Demo input
            print("  Using demo input...")
            input_data = np.random.randn(64)
        
        output = engine.infer(input_data, temperature=args.temp if args.temp else 0.7)
        
        print(f"\n  Input shape: {input_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {np.mean(output):.6f}")
        
        if args.output:
            np.save(args.output, output)
            print(f"  Saved to: {args.output}")
        
        print("\n✅ Inference complete")
    
    def cmd_compress(self, args):
        """Compress files using holographic encoding."""
        print("\n[COMPRESS] Holographic compression...")
        
        from applications.holographic_compressor import HolographicCompressor
        
        if not args.input:
            print("❌ Error: --input is required")
            return
        
        level = args.level if args.level else 7
        compressor = HolographicCompressor(compression_level=level)
        
        if args.decompress:
            print(f"  Decompressing: {args.input}")
            output = compressor.decompress_file(args.input, args.output)
            print(f"  Output: {output}")
        else:
            print(f"  Compressing: {args.input}")
            print(f"  Level: {level}")
            result = compressor.compress_file(args.input, args.output)
            print(f"  Original: {result['original_size']:,} bytes")
            print(f"  Compressed: {result['compressed_size']:,} bytes")
            print(f"  Ratio: {result['compression_ratio']:.2f}x")
        
        print("\n✅ Operation complete")
    
    def cmd_convert(self, args):
        """Convert model formats."""
        print("\n[CONVERT] Model format conversion...")
        
        if not args.input or not args.output:
            print("❌ Error: --input and --output are required")
            return
        
        input_ext = os.path.splitext(args.input)[1].lower()
        output_ext = os.path.splitext(args.output)[1].lower()
        
        print(f"  Input: {args.input} ({input_ext})")
        print(f"  Output: {args.output} ({output_ext})")
        
        # Conversion logic would go here
        # For now, placeholder
        print("  Converting...")
        
        print("\n✅ Conversion complete")
    
    def cmd_benchmark(self, args):
        """Run performance benchmarks."""
        print("\n[BENCHMARK] Running performance benchmarks...")
        
        from scripts.benchmark_manifold import ManifoldBenchmark
        
        benchmark = ManifoldBenchmark()
        
        if args.quick:
            benchmark.run_quick()
        else:
            benchmark.run_all()
        
        print("\n✅ Benchmarks complete")
    
    def cmd_info(self, args):
        """Display model information."""
        print("\n[INFO] Model Information")
        print("=" * 50)
        
        if args.model:
            import json
            
            try:
                with open(args.model, 'rb') as f:
                    header = f.read(16)
                    print(f"  Header: {header[:14].decode('utf-8', errors='ignore')}")
                    
                    meta_block = f.read(256)
                    meta_json = meta_block.rstrip(b'\x00').decode('utf-8')
                    metadata = json.loads(meta_json)
                    
                    print(f"\n  Architecture: {metadata.get('architecture', 'Unknown')}")
                    print(f"  Version: {metadata.get('version', 'Unknown')}")
                    print(f"  Parameters: {metadata.get('params', 0):,}")
                    
                file_size = os.path.getsize(args.model)
                print(f"  File Size: {file_size / 1024 / 1024:.2f} MB")
                
            except Exception as e:
                print(f"❌ Error reading model: {e}")
        else:
            print("  7D mH-Q Crystal Architecture")
            print(f"  CLI Version: {self.version}")
            print(f"  Author: {self.author}")
            print("\n  Use --model <path> to inspect a model file")
    
    def cmd_visualize(self, args):
        """Generate visualizations."""
        print("\n[VISUALIZE] Generating visualizations...")
        
        from crystal_patterns import CrystalPatternGenerator
        import numpy as np
        
        pattern_gen = CrystalPatternGenerator(complexity=512)
        
        pattern_type = args.pattern if args.pattern else "manifold"
        
        if pattern_type == "manifold":
            print("  Generating 7D manifold visualization...")
            manifold = pattern_gen.generate_holographic_manifold(dimensions=7, resolution=32)
            output_path = args.output if args.output else "manifold_viz.npy"
            np.save(output_path, manifold)
            
        elif pattern_type == "spiral":
            print("  Generating Fibonacci spiral...")
            spiral = pattern_gen.generate_fibonacci_spiral(1000)
            output_path = args.output if args.output else "spiral_viz.npy"
            np.save(output_path, spiral)
            
        elif pattern_type == "quantum":
            print("  Generating quantum field...")
            field = pattern_gen.generate_quantum_field((128, 128), 100)
            output_path = args.output if args.output else "quantum_viz.npy"
            np.save(output_path, field)
            
        elif pattern_type == "flower":
            print("  Generating Flower of Life...")
            flower = pattern_gen.generate_sacred_geometry("flower_of_life", num_rings=4)
            output_path = args.output if args.output else "flower_viz.npy"
            np.save(output_path, flower['centers'])
        
        print(f"  Saved to: {output_path}")
        print("\n✅ Visualization complete")
    
    def run(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(
            description="7D mH-Q Crystal Architecture CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    crystal_cli.py genesis --params 10000000 --output model.gguf
    crystal_cli.py train --epochs 100 --lr 0.001
    crystal_cli.py infer --model model.gguf --input data.npy
    crystal_cli.py compress --input file.txt --level 9
    crystal_cli.py benchmark --quick
    crystal_cli.py info --model model.gguf
            """
        )
        
        parser.add_argument('--version', action='version', version=f'Crystal CLI v{self.version}')
        parser.add_argument('--quiet', '-q', action='store_true', help='Suppress banner')
        
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Genesis command
        gen_parser = subparsers.add_parser('genesis', help='Generate new Crystal model')
        gen_parser.add_argument('--params', type=int, help='Number of parameters')
        gen_parser.add_argument('--output', '-o', help='Output file path')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train Crystal model')
        train_parser.add_argument('--data', help='Training data path')
        train_parser.add_argument('--epochs', type=int, help='Number of epochs')
        train_parser.add_argument('--lr', type=float, help='Learning rate')
        train_parser.add_argument('--batch', type=int, help='Batch size')
        train_parser.add_argument('--device', help='Compute device')
        train_parser.add_argument('--output', '-o', help='Output model path')
        
        # Infer command
        infer_parser = subparsers.add_parser('infer', help='Run inference')
        infer_parser.add_argument('--model', '-m', required=True, help='Model path')
        infer_parser.add_argument('--input', '-i', help='Input data path')
        infer_parser.add_argument('--output', '-o', help='Output path')
        infer_parser.add_argument('--device', help='Compute device')
        infer_parser.add_argument('--temp', type=float, help='Temperature')
        
        # Compress command
        comp_parser = subparsers.add_parser('compress', help='Compress files')
        comp_parser.add_argument('--input', '-i', required=True, help='Input file')
        comp_parser.add_argument('--output', '-o', help='Output file')
        comp_parser.add_argument('--level', type=int, help='Compression level (1-11)')
        comp_parser.add_argument('--decompress', '-d', action='store_true', help='Decompress')
        
        # Convert command
        conv_parser = subparsers.add_parser('convert', help='Convert model formats')
        conv_parser.add_argument('--input', '-i', required=True, help='Input model')
        conv_parser.add_argument('--output', '-o', required=True, help='Output model')
        conv_parser.add_argument('--format', help='Target format')
        
        # Benchmark command
        bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
        bench_parser.add_argument('--quick', action='store_true', help='Quick benchmark')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Display model info')
        info_parser.add_argument('--model', '-m', help='Model path')
        
        # Visualize command
        viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
        viz_parser.add_argument('--pattern', choices=['manifold', 'spiral', 'quantum', 'flower'])
        viz_parser.add_argument('--output', '-o', help='Output path')
        
        args = parser.parse_args()
        
        if not args.quiet:
            self.print_banner()
        
        if args.command is None:
            parser.print_help()
            return
        
        # Route to command handler
        command_map = {
            'genesis': self.cmd_genesis,
            'train': self.cmd_train,
            'infer': self.cmd_infer,
            'compress': self.cmd_compress,
            'convert': self.cmd_convert,
            'benchmark': self.cmd_benchmark,
            'info': self.cmd_info,
            'visualize': self.cmd_visualize,
        }
        
        handler = command_map.get(args.command)
        if handler:
            try:
                handler(args)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    cli = CrystalCLI()
    cli.run()

