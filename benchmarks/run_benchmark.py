#!/usr/bin/env python3
"""
Sloth Benchmark CLI

Usage:
    python benchmarks/run_benchmark.py add
    python benchmarks/run_benchmark.py add --impls sloth-triton,torch
    python benchmarks/run_benchmark.py --all
    python benchmarks/run_benchmark.py add --output reports/add.json --format json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.compare.benchmark_compare import BenchmarkComparator


def main():
    parser = argparse.ArgumentParser(description="Sloth CUDA Operators Benchmark")
    parser.add_argument("operator", nargs="?", help="Operator to benchmark (e.g., add, softmax, gemm)")
    parser.add_argument("--all", action="store_true", help="Run all operators")
    parser.add_argument("--impls", type=str, help="Implementations to test (comma-separated)")
    parser.add_argument("--output", type=str, default="reports/benchmark.json", help="Output file path")
    parser.add_argument("--format", choices=["json", "markdown", "table"], default="table", help="Output format")

    args = parser.parse_args()

    if not args.operator and not args.all:
        parser.print_help()
        print("\nAvailable operators:")
        comparator = BenchmarkComparator()
        for op in comparator.config.get("operators", {}).keys():
            print(f"  - {op}")
        sys.exit(1)

    comparator = BenchmarkComparator()

    implementations = None
    if args.impls:
        implementations = [x.strip() for x in args.impls.split(",")]

    if args.all:
        all_results = []
        for op in comparator.config.get("operators", {}).keys():
            print(f"\n{'=' * 60}")
            print(f"Running benchmarks for {op}...")
            results = comparator.run_all_workloads(op, implementations)
            all_results.extend(results)

        comparator.print_summary(all_results)

        if args.format in ["json", "markdown"]:
            comparator.save_report(all_results, args.output, args.format)
    else:
        results = comparator.run_all_workloads(args.operator, implementations)

        for result in results:
            comparator.print_report(result)

        comparator.print_summary(results)

        if args.format in ["json", "markdown"]:
            comparator.save_report(results, args.output, args.format)


if __name__ == "__main__":
    main()
