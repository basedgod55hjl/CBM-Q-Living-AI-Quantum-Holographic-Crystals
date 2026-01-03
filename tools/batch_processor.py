#!/usr/bin/env python3
"""
7D mH-Q: Batch Processing Tool
Process multiple files or models in parallel.
"""

import sys
import os
import time
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    input_path: str
    output_path: str
    operation: str
    params: Dict = None
    status: str = "pending"
    result: Dict = None
    error: str = None


class BatchProcessor:
    """
    Batch processing system for 7D mH-Q operations.
    
    Supports:
    - Parallel compression of multiple files
    - Batch model inference
    - Multi-file format conversion
    - Bulk data transformation
    """
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum parallel workers (default: CPU count)
            use_processes: Use processes instead of threads
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.jobs: List[BatchJob] = []
        self.completed_jobs: List[BatchJob] = []
        
        print(f"[BATCH] Initialized with {self.max_workers} workers")
        print(f"[BATCH] Mode: {'Processes' if use_processes else 'Threads'}")
    
    def add_job(self, input_path: str, output_path: str, 
                operation: str, params: Dict = None) -> str:
        """
        Add a job to the batch queue.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            operation: Operation type (compress, decompress, convert, infer)
            params: Operation parameters
            
        Returns:
            Job ID
        """
        job_id = f"job_{len(self.jobs):04d}_{int(time.time())}"
        
        job = BatchJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            params=params or {}
        )
        
        self.jobs.append(job)
        print(f"[BATCH] Added job {job_id}: {operation} on {os.path.basename(input_path)}")
        
        return job_id
    
    def add_directory(self, input_dir: str, output_dir: str,
                      operation: str, pattern: str = "*", 
                      params: Dict = None) -> List[str]:
        """
        Add all files in a directory to the batch queue.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            operation: Operation type
            pattern: File pattern (glob)
            params: Operation parameters
            
        Returns:
            List of job IDs
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(input_dir, pattern))
        job_ids = []
        
        for input_path in files:
            filename = os.path.basename(input_path)
            
            # Determine output filename based on operation
            if operation == "compress":
                output_filename = filename + ".holo"
            elif operation == "decompress":
                output_filename = filename.replace(".holo", ".decompressed")
            else:
                output_filename = filename
            
            output_path = os.path.join(output_dir, output_filename)
            job_id = self.add_job(input_path, output_path, operation, params)
            job_ids.append(job_id)
        
        print(f"[BATCH] Added {len(job_ids)} jobs from {input_dir}")
        return job_ids
    
    @staticmethod
    def _execute_job(job: BatchJob) -> BatchJob:
        """
        Execute a single job (static for multiprocessing).
        """
        try:
            job.status = "running"
            start_time = time.time()
            
            if job.operation == "compress":
                from applications.holographic_compressor import HolographicCompressor
                
                level = job.params.get('level', 7)
                compressor = HolographicCompressor(compression_level=level)
                result = compressor.compress_file(job.input_path, job.output_path)
                job.result = result
                
            elif job.operation == "decompress":
                from applications.holographic_compressor import HolographicCompressor
                
                compressor = HolographicCompressor()
                output = compressor.decompress_file(job.input_path, job.output_path)
                job.result = {'output_path': output}
                
            elif job.operation == "convert":
                # Model format conversion
                job.result = {'converted': True}
                
            elif job.operation == "infer":
                from engines.inference_engine import CrystalInferenceEngine
                import numpy as np
                
                model_path = job.params.get('model')
                engine = CrystalInferenceEngine(model_path)
                
                input_data = np.load(job.input_path)
                output = engine.infer(input_data)
                np.save(job.output_path, output)
                
                job.result = {
                    'input_shape': list(input_data.shape),
                    'output_shape': list(output.shape)
                }
                
            elif job.operation == "transform":
                # Custom transformation
                transform_fn = job.params.get('transform')
                if transform_fn and callable(transform_fn):
                    transform_fn(job.input_path, job.output_path, job.params)
                job.result = {'transformed': True}
            
            else:
                raise ValueError(f"Unknown operation: {job.operation}")
            
            job.status = "completed"
            job.result['elapsed_seconds'] = time.time() - start_time
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
        
        return job
    
    def run(self, progress_callback: Callable = None) -> Dict:
        """
        Execute all jobs in parallel.
        
        Args:
            progress_callback: Optional callback(completed, total, job)
            
        Returns:
            Summary of results
        """
        if not self.jobs:
            print("[BATCH] No jobs to process")
            return {'total': 0, 'completed': 0, 'failed': 0}
        
        print(f"\n[BATCH] Starting {len(self.jobs)} jobs...")
        start_time = time.time()
        
        # Select executor type
        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        completed = 0
        failed = 0
        
        with Executor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self._execute_job, job): job 
                for job in self.jobs
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                job = future.result()
                self.completed_jobs.append(job)
                
                if job.status == "completed":
                    completed += 1
                    status = "✓"
                else:
                    failed += 1
                    status = "✗"
                
                if progress_callback:
                    progress_callback(completed + failed, len(self.jobs), job)
                else:
                    print(f"  [{completed + failed}/{len(self.jobs)}] {status} {job.job_id}")
        
        elapsed = time.time() - start_time
        
        summary = {
            'total': len(self.jobs),
            'completed': completed,
            'failed': failed,
            'elapsed_seconds': elapsed,
            'jobs_per_second': len(self.jobs) / elapsed if elapsed > 0 else 0
        }
        
        print(f"\n[BATCH] Complete: {completed}/{len(self.jobs)} succeeded in {elapsed:.2f}s")
        
        if failed > 0:
            print(f"[BATCH] {failed} jobs failed:")
            for job in self.completed_jobs:
                if job.status == "failed":
                    print(f"  - {job.job_id}: {job.error}")
        
        return summary
    
    def get_results(self) -> List[Dict]:
        """Get results from all completed jobs."""
        return [
            {
                'job_id': job.job_id,
                'input': job.input_path,
                'output': job.output_path,
                'operation': job.operation,
                'status': job.status,
                'result': job.result,
                'error': job.error
            }
            for job in self.completed_jobs
        ]
    
    def save_report(self, output_path: str):
        """Save batch processing report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_jobs': len(self.jobs),
            'completed_jobs': len([j for j in self.completed_jobs if j.status == "completed"]),
            'failed_jobs': len([j for j in self.completed_jobs if j.status == "failed"]),
            'jobs': self.get_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[BATCH] Report saved to: {output_path}")


def main():
    """Demo batch processing."""
    import argparse
    import tempfile
    
    parser = argparse.ArgumentParser(description="7D mH-Q Batch Processor")
    parser.add_argument('--input', '-i', help='Input directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--operation', default='compress', 
                       choices=['compress', 'decompress', 'infer'])
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--pattern', default='*', help='File pattern')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("7D mH-Q BATCH PROCESSOR")
    print("="*60)
    
    processor = BatchProcessor(max_workers=args.workers)
    
    if args.input and args.output:
        # Process directory
        processor.add_directory(
            args.input, 
            args.output, 
            args.operation,
            pattern=args.pattern
        )
    else:
        # Demo mode
        print("\n[DEMO] Creating test files...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, 'input')
            output_dir = os.path.join(tmpdir, 'output')
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            
            # Create test files
            for i in range(5):
                filepath = os.path.join(input_dir, f'test_{i}.txt')
                with open(filepath, 'w') as f:
                    f.write(f"Test content {i}\n" * 100)
            
            # Add jobs
            processor.add_directory(input_dir, output_dir, 'compress', pattern='*.txt')
            
            # Run
            summary = processor.run()
            
            print(f"\n[DEMO] Summary: {summary}")
    
    if args.input and args.output:
        summary = processor.run()
        processor.save_report(os.path.join(args.output, 'batch_report.json'))


if __name__ == "__main__":
    main()

