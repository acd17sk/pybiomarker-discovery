#!/usr/bin/env python3
"""
Test runner for text biomarker model tests.
Provides convenient interface for running tests with various options.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --coverage         # With coverage report
    python run_tests.py --quick            # Quick smoke test
    python run_tests.py --gpu              # GPU tests only
    python run_tests.py -f test_linguistic_analyzer.py
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_test_directory():
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def run_tests(args):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Always use the current directory (tests/models/text/)
    test_dir = get_test_directory()
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    elif not args.quiet:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=biomarkers.models.text",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Build marker filter
    markers = []
    if args.fast:
        markers.append("not slow")
    if args.no_gpu:
        markers.append("not gpu")
    if args.gpu:
        markers.append("gpu")
    if args.integration:
        markers.append("integration")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add specific test file
    if args.file:
        test_file = test_dir / args.file
        if not test_file.exists():
            print(f"❌ Error: Test file not found: {test_file}")
            return 1
        cmd.append(str(test_file))
    else:
        # Run all tests in this directory
        cmd.append(str(test_dir))
    
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
    
    # Force CPU
    if args.force_cpu:
        import os
        os.environ['PYTEST_FORCE_CPU'] = '1'
        print("🖥️  Forcing CPU mode (CUDA disabled)")
    
    # Additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    # Print command
    print("=" * 70)
    print("Text Biomarker Model Test Runner")
    print("=" * 70)
    print("Running:", " ".join(cmd))
    print("-" * 70)
    
    # Run tests
    try:
        # Use check=False to handle non-zero exit codes gracefully
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1


def list_tests(args):
    """List all available tests."""
    test_dir = get_test_directory()
    
    cmd = ["python", "-m", "pytest", str(test_dir), "--collect-only", "-q"]
    
    if args.file:
        test_file = test_dir / args.file
        if not test_file.exists():
            print(f"❌ Error: Test file not found: {test_file}")
            return 1
        cmd = ["python", "-m", "pytest", str(test_file), "--collect-only", "-q"]
    
    print("=" * 70)
    print(f"Available Tests (in {test_dir.name if not args.file else args.file})")
    print("=" * 70)
    subprocess.run(cmd)


def run_quick_check():
    """Run quick smoke tests to verify basic functionality."""
    test_dir = get_test_directory()
    
    print("=" * 70)
    print("Running Quick Smoke Tests")
    print("=" * 70)
    print("Testing basic initialization and imports...")
    print("-" * 70)
    
    # Key tests to verify everything works
    tests = [
        "test_linguistic_analyzer.py::TestLexicalDiversityAnalyzer::test_initialization",
        "test_linguistic_analyzer.py::TestSyntacticComplexityAnalyzer::test_initialization",
        "test_linguistic_analyzer.py::TestLinguisticAnalyzer::test_initialization",
        "test_text_biomarker.py::TestTextBiomarkerModelInitialization::test_basic_initialization",
    ]
    
    # Build full paths for pytest
    test_paths = []
    for t in tests:
        file_name, test_path = t.split("::", 1)
        full_file_path = test_dir / file_name
        if not full_file_path.exists():
            print(f"⚠️  Warning: Quick check test file not found: {file_name}")
            continue
        test_paths.append(f"{full_file_path}::{test_path}")

    if not test_paths:
        print("❌ Error: No quick check test paths found.")
        return 1

    cmd = ["python", "-m", "pytest", "-v", "--tb=short"] + test_paths
    
    result = subprocess.run(cmd, check=False)
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✅ Quick check passed! All modules are working.")
        print("\nYou can now run:")
        print("  python run_tests.py          # All tests")
        print("  python run_tests.py --full   # Full suite with coverage")
    else:
        print("❌ Quick check failed. Please review errors above.")
        print("\nTry running:")
        print("  python run_tests.py --verbose")
    print("=" * 70)
    
    return result.returncode


def run_full_suite():
    """Run complete test suite with coverage."""
    test_dir = get_test_directory()
    
    print("=" * 70)
    print("Running Full Test Suite")
    print("=" * 70)
    print("This will run all tests with coverage reporting...")
    print("-" * 70)
    
    cmd = [
        "python", "-m", "pytest",
        str(test_dir),
        "-v",
        "--cov=biomarkers.models.text",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10",
        "--tb=short"
    ]
    
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✅ Full test suite passed!")
        print("\n📊 Coverage report generated:")
        print(f"  Open: {Path.cwd() / 'htmlcov' / 'index.html'}")
    else:
        print("❌ Some tests failed. Please review errors above.")
    print("=" * 70)
    
    return result.returncode


def run_component_tests():
    """Run tests organized by component."""
    test_dir = get_test_directory()
    
    components = {
        "Lexical Analysis": "test_linguistic_analyzer.py::TestLexicalDiversityAnalyzer",
        "Syntactic Analysis": "test_linguistic_analyzer.py::TestSyntacticComplexityAnalyzer",
        "Semantic Analysis": "test_linguistic_analyzer.py::TestSemanticCoherenceAnalyzer",
        "Discourse Analysis": "test_linguistic_analyzer.py::TestDiscourseStructureAnalyzer",
        "Cognitive Load": "test_linguistic_analyzer.py::TestCognitiveLoadAnalyzer",
        "Decline Markers": "test_linguistic_analyzer.py::TestLinguisticDeclineAnalyzer",
        "Temporal Analysis": "test_linguistic_analyzer.py::TestTemporalAnalyzer",
        "Full Model (Forward Pass)": "test_text_biomarker.py::TestForwardPass",
        "Full Model (Initialization)": "test_text_biomarker.py::TestTextBiomarkerModelInitialization",
    }
    
    print("=" * 70)
    print("Component Test Menu")
    print("=" * 70)
    print("\nAvailable components:")
    component_keys = list(components.keys())
    for i, name in enumerate(component_keys, 1):
        print(f"  {i}. {name}")
    print("  0. Run all components")
    print("\n" + "-" * 70)
    
    try:
        choice = input(f"Select component to test (0-{len(component_keys)}): ").strip()
        
        if choice == "0":
            # Run all
            cmd = ["python", "-m", "pytest", str(test_dir), "-v"]
            print(f"\n🧪 Testing: All Components")
            print("-" * 70)
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)
        elif choice.isdigit() and 1 <= int(choice) <= len(components):
            # Run selected component
            component_name = component_keys[int(choice) - 1]
            component_path = components[component_name]
            
            # Construct full path
            file_name, test_path = component_path.split("::", 1)
            full_test_path = str(test_dir / file_name) + "::" + test_path
            
            print(f"\n🧪 Testing: {component_name}")
            print("-" * 70)
            
            cmd = ["python", "-m", "pytest", full_test_path, "-v"]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)
        else:
            print("Invalid choice")
            return 1
    except KeyboardInterrupt:
        print("\n\nCancelled")
        return 1
    
    return 0


def show_stats():
    """Show test statistics by collecting tests."""
    test_dir = get_test_directory()
    
    print("=" * 70)
    print("Test Suite Statistics")
    print("=" * 70)
    
    # Count tests
    cmd = ["python", "-m", "pytest", str(test_dir), "--collect-only", "-q"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Failed to collect tests.")
        print(e.stderr)
        return 1
    
    lines = result.stdout.strip().split('\n')
    test_count = 0
    test_files = set()
    
    # FIXED: More robust parsing
    for line in lines:
        if '::' in line:  # A better indicator of a test function/method
            test_count += 1
            try:
                # Get the file path relative to the test_dir
                full_path = Path(line.split('::')[0])
                relative_path = full_path.relative_to(test_dir.parent)
                test_files.add(str(relative_path))
            except ValueError:
                test_files.add(line.split('::')[0])
    
    print(f"\n📊 Test Files ({len(test_files)}):")
    for f in sorted(list(test_files)):
        print(f"  - {f}")
    
    print(f"\n📈 Total Tests Collected: {test_count}")
    
    # FIXED: Removed hardcoded, out-of-date statistics
    print(f"\n🧪 Test Categories:")
    print(f"  - Use 'python run_tests.py --components' for an interactive menu.")
    
    print("\n" + "=" * 70)
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for text biomarker models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                     # Run all tests
  python run_tests.py --quick             # Quick smoke test
  python run_tests.py --full              # Full suite with coverage
  python run_tests.py --fast              # Skip slow tests
  python run_tests.py --coverage          # Run with coverage
  python run_tests.py --force-cpu         # Force CPU mode
  python run_tests.py -f test_linguistic_analyzer.py
  python run_tests.py -t test_initialization
  python run_tests.py --list              # List all tests
  python run_tests.py --stats             # Show statistics
  python run_tests.py --components        # Component menu
  python run_tests.py -n 4                # Run with 4 workers
        """
    )
    
    # Special modes
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick smoke tests (verify basic functionality)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full test suite with coverage'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available tests'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show test statistics'
    )
    parser.add_argument(
        '--components',
        action='store_true',
        help='Interactive component test menu'
    )
    
    # Test selection
    parser.add_argument(
        '--file', '-f',
        help='Run specific test file (e.g., test_linguistic_analyzer.py)'
    )
    parser.add_argument(
        '--test', '-k', # Changed from -t to -k to match pytest
        help='Run tests matching pattern (e.g., test_initialization)'
    )
    
    # Test filtering
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests (use -m "not slow")'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Run only GPU tests (use -m "gpu")'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Skip GPU tests (use -m "not gpu")'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU mode (set PYTEST_FORCE_CPU=1)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (-vv)'
    )
    parser.add_argument(
        '--quiet', '-q', # Matched -q to --quick
        action='store_true',
        help='Minimal output (opposite of verbose)'
    )
    parser.add_argument(
        '--capture', '-s',
        action='store_true',
        help='Show print statements (disable output capture)'
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
        type=str, # Allow 'auto'
        metavar='WORKERS',
        help='Run tests in parallel with N workers (e.g., -n 4 or -n auto)'
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
        help='Additional arguments to pass to pytest (e.g., --pytest-args="--lf --sw")'
    )
    
    args = parser.parse_args()
    
    # Handle special modes first
    if args.list:
        return list_tests(args)
    
    if args.stats:
        return show_stats()
    
    if args.components:
        return run_component_tests()
    
    if args.quick:
        return run_quick_check()
    
    if args.full:
        return run_full_suite()
    
    # Run tests with options
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())