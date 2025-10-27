#!/usr/bin/env python
"""
Test runner for discovery module tests
Provides convenient interface for running tests with various options
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    elif not args.quiet:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=discovery", "--cov-report=html", "--cov-report=term"])
    
    # Add markers
    markers = []
    if args.fast:
        markers.append("not slow")
    if args.no_gpu:
        markers.append("not gpu")
    if args.integration:
        markers.append("integration")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add specific test file
    if args.file:
        cmd.append(args.file)
    
    # Add test name filter
    if args.test:
        cmd.extend(["-k", args.test])
    
    # Add durations
    if args.durations:
        cmd.extend(["--durations", str(args.durations)])
    
    # Add stop on first failure
    if args.exitfirst:
        cmd.append("-x")
    
    # Add pdb on failure
    if args.pdb:
        cmd.append("--pdb")
    
    # Show output
    if args.capture:
        cmd.append("-s")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    # Print command
    print("Running:", " ".join(cmd))
    print("-" * 70)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1


def list_tests(args):
    """List all available tests."""
    cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
    
    if args.file:
        cmd.append(args.file)
    
    subprocess.run(cmd)


def run_quick_check():
    """Run quick smoke tests to verify basic functionality."""
    print("Running quick smoke tests...")
    print("=" * 70)
    
    tests = [
        "test_feature_discovery.py::TestAutomatedFeatureDiscovery::test_initialization",
        "test_neural_architecture_search.py::TestNeuralArchitectureSearch::test_initialization",
        "test_contrastive_learner.py::TestContrastiveBiomarkerLearner::test_initialization_simclr"
    ]
    
    cmd = ["python", "-m", "pytest", "-v"] + tests
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✓ Quick check passed! All modules are working.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ Quick check failed. Please review errors above.")
        print("=" * 70)
    
    return result.returncode


def run_full_suite():
    """Run complete test suite with coverage."""
    print("Running full test suite...")
    print("=" * 70)
    
    cmd = [
        "python", "-m", "pytest",
        "-v",
        "--cov=discovery",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10"
    ]
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✓ Full test suite passed!")
        print("Coverage report generated in htmlcov/index.html")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ Some tests failed. Please review errors above.")
        print("=" * 70)
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for discovery module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s --fast                   # Run fast tests only
  %(prog)s --coverage               # Run with coverage
  %(prog)s --file test_feature_discovery.py
  %(prog)s --test test_initialization
  %(prog)s --quick                  # Quick smoke test
  %(prog)s --full                   # Full suite with coverage
  %(prog)s --list                   # List all tests
        """
    )
    
    # Test selection
    parser.add_argument(
        '--file', '-f',
        help='Run specific test file'
    )
    parser.add_argument(
        '--test', '-t',
        help='Run tests matching pattern'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available tests'
    )
    
    # Quick modes
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick smoke tests'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full test suite with coverage'
    )
    
    # Test filtering
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Skip GPU tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--capture', '-s',
        action='store_true',
        help='Show print statements'
    )
    
    # Coverage
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Run with coverage report'
    )
    
    # Performance
    parser.add_argument(
        '--durations',
        type=int,
        metavar='N',
        help='Show N slowest tests'
    )
    parser.add_argument(
        '--parallel', '-n',
        type=int,
        metavar='WORKERS',
        help='Run tests in parallel with N workers'
    )
    
    # Debugging
    parser.add_argument(
        '--pdb',
        action='store_true',
        help='Drop into debugger on failure'
    )
    parser.add_argument(
        '--exitfirst', '-x',
        action='store_true',
        help='Stop on first failure'
    )
    
    # Pass-through args
    parser.add_argument(
        '--pytest-args',
        help='Additional arguments to pass to pytest'
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.list:
        return list_tests(args)
    
    if args.quick:
        return run_quick_check()
    
    if args.full:
        return run_full_suite()
    
    # Run tests with options
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())