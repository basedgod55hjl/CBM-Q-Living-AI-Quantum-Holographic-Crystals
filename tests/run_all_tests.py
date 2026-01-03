#!/usr/bin/env python3
"""
7D mH-Q: Master Test Runner
Executes all test suites and generates comprehensive report.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner():
    """Print test banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ████████╗ ██████╗     ███╗   ███╗██╗  ██╗       ██████╗       ║
║   ╚════██║ ██╔══██╗    ████╗ ████║██║  ██║      ██╔═══██╗      ║
║       ██╔╝ ██║  ██║    ██╔████╔██║███████║█████╗██║   ██║      ║
║      ██╔╝  ██║  ██║    ██║╚██╔╝██║██╔══██║╚════╝██║▄▄ ██║      ║
║      ██║   ██████╔╝    ██║ ╚═╝ ██║██║  ██║      ╚██████╔╝      ║
║      ╚═╝   ╚═════╝     ╚═╝     ╚═╝╚═╝  ╚═╝       ╚══▀▀═╝       ║
║                                                                  ║
║   COMPREHENSIVE TEST SUITE                                       ║
║   Crystal Architecture Verification                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_test_suite(name: str, test_module):
    """Run a test suite and return results."""
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {name}")
    print(f"{'#'*70}")
    
    try:
        tester = test_module()
        results = tester.run_all_tests()
        return {
            'name': name,
            'success': True,
            'results': results
        }
    except Exception as e:
        print(f"\n❌ ERROR in {name}: {e}")
        return {
            'name': name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main test runner."""
    print_banner()
    
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    all_results = []
    start_time = time.time()
    
    # Import and run test suites
    test_suites = []
    
    # Stability tests
    try:
        from test_stability import StabilityTester
        test_suites.append(("Stability (S²) Tests", StabilityTester))
    except ImportError as e:
        print(f"⚠️ Could not import stability tests: {e}")
    
    # Convergence tests
    try:
        from test_convergence import ConvergenceTester
        test_suites.append(("Convergence Tests", ConvergenceTester))
    except ImportError as e:
        print(f"⚠️ Could not import convergence tests: {e}")
    
    # Compression tests
    try:
        from test_compression import CompressionTester
        test_suites.append(("Compression Tests", CompressionTester))
    except ImportError as e:
        print(f"⚠️ Could not import compression tests: {e}")
    
    # Run all suites
    for name, tester_class in test_suites:
        result = run_test_suite(name, tester_class)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total_passed = 0
    total_tests = 0
    
    for suite in all_results:
        name = suite['name']
        if suite['success']:
            results = suite['results']
            passed = results['passed']
            total = results['total']
            total_passed += passed
            total_tests += total
            status = "[PASS]" if results['all_passed'] else "[WARN]"
            print(f"  {status} {name}: {passed}/{total} passed")
        else:
            print(f"  ❌ {name}: FAILED TO RUN - {suite.get('error', 'Unknown error')}")
    
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    print(f"  Time: {total_time:.2f} seconds")
    
    # Final verdict
    all_passed = total_passed == total_tests and all(s['success'] for s in all_results)
    
    if all_passed:
        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED - 7D mH-Q ARCHITECTURE VERIFIED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("[WARNING] SOME TESTS FAILED - REVIEW RESULTS ABOVE")
        print("="*70)
    
    # Save results to file
    results_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_results.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_passed': total_passed,
            'total_tests': total_tests,
            'all_passed': all_passed,
            'elapsed_seconds': total_time,
            'suites': all_results
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

