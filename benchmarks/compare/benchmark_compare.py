import torch
import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from runners import get_runner


class BenchmarkComparator:
    def __init__(self, config_path: str = "benchmarks/configs/workload.yaml"):
        self.console = Console()
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _create_inputs(self, operator: str, workload: Dict, device: str = "cuda") -> Dict[str, torch.Tensor]:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(workload.get("dtype", "float32"), torch.float32)

        torch.manual_seed(self.config.get("global", {}).get("seed", 42))

        if operator == "add":
            shape = workload["shape"]
            return {
                "a": torch.randn(shape, dtype=dtype, device=device),
                "b": torch.randn(shape, dtype=dtype, device=device),
            }
        elif operator == "softmax":
            shape = workload["shape"]
            return {
                "x": torch.randn(shape, dtype=dtype, device=device),
            }
        elif operator == "gemm":
            M, K, N = workload["M"], workload["K"], workload["N"]
            return {
                "a": torch.randn(M, K, dtype=dtype, device=device),
                "b": torch.randn(K, N, dtype=dtype, device=device),
            }
        else:
            shape = workload.get("shape", [1024, 1024])
            return {
                "a": torch.randn(shape, dtype=dtype, device=device),
                "b": torch.randn(shape, dtype=dtype, device=device),
            }

    def run_single_workload(
        self,
        operator: str,
        workload_name: str,
        implementations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        operator_config = self.config["operators"].get(operator)
        if not operator_config:
            raise ValueError(f"Operator {operator} not found in config")

        workload = None
        for w in operator_config["workloads"]:
            if w["name"] == workload_name:
                workload = w
                break

        if not workload:
            raise ValueError(f"Workload {workload_name} not found for operator {operator}")

        implementations = implementations or operator_config.get("implementations", ["torch"])
        inputs = self._create_inputs(operator, workload)

        results = {}
        torch_time = None

        for impl in implementations:
            try:
                runner = get_runner(impl, operator)
                result = runner.benchmark(
                    inputs,
                    warmup=workload.get("warmup", 10),
                    repeat=workload.get("repeat", 100),
                )
                results[impl] = result

                if impl == "torch":
                    torch_time = result["mean_ms"]

            except Exception as e:
                self.console.print(f"[red]Error running {impl}: {e}[/red]")
                results[impl] = {"error": str(e)}

        if torch_time:
            for impl, result in results.items():
                if "mean_ms" in result:
                    result["speedup_vs_torch"] = torch_time / result["mean_ms"]

        return {
            "operator": operator,
            "workload": workload_name,
            "workload_config": workload,
            "results": results,
        }

    def run_all_workloads(self, operator: str, implementations: Optional[List[str]] = None) -> List[Dict]:
        operator_config = self.config["operators"].get(operator)
        if not operator_config:
            raise ValueError(f"Operator {operator} not found in config")

        all_results = []
        for workload in operator_config["workloads"]:
            result = self.run_single_workload(operator, workload["name"], implementations)
            all_results.append(result)

        return all_results

    def print_report(self, results: Dict, format: str = "table"):
        self.console.print(Panel(f"[bold cyan]BENCHMARK REPORT: {results['operator']}[/bold cyan]"))

        workload_info = results["workload_config"]
        shape_str = str(workload_info.get("shape", "N/A"))
        dtype_str = workload_info.get("dtype", "float32")

        self.console.print(f"\n[bold]Workload: {results['workload']}[/bold] ({shape_str}, {dtype_str})")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Implementation", style="cyan")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Speedup vs Torch", justify="right")

        for impl, result in results["results"].items():
            if "error" in result:
                table.add_row(impl, "ERROR", "N/A")
            else:
                time_str = f"{result['mean_ms']:.3f}"
                speedup_str = (
                    f"{result.get('speedup_vs_torch', 1.0):.2f}x" if "speedup_vs_torch" in result else "baseline"
                )
                table.add_row(impl, time_str, speedup_str)

        self.console.print(table)

    def print_summary(self, all_results: List[Dict]):
        self.console.print("\n" + "=" * 80)
        self.console.print(Panel("[bold yellow]SUMMARY (Geometric Mean Speedup vs Torch)[/bold yellow]"))

        impl_speedups = {}
        for result in all_results:
            for impl, r in result["results"].items():
                if "speedup_vs_torch" in r and r["speedup_vs_torch"] > 0:
                    if impl not in impl_speedups:
                        impl_speedups[impl] = []
                    impl_speedups[impl].append(r["speedup_vs_torch"])

        summary_table = Table(show_header=True, header_style="bold green")
        summary_table.add_column("Implementation", style="cyan")
        summary_table.add_column("Geometric Mean Speedup", justify="right")

        for impl, speedups in sorted(impl_speedups.items(), key=lambda x: -np.prod(x[1])):
            if speedups:
                geo_mean = np.exp(np.mean(np.log(speedups)))
                summary_table.add_row(impl, f"{geo_mean:.2f}x")

        self.console.print(summary_table)

    def save_report(self, results: List[Dict], output_path: str, format: str = "json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif format == "markdown":
            self._save_markdown(results, output_path)

        self.console.print(f"[green]Report saved to {output_path}[/green]")

    def _save_markdown(self, results: List[Dict], output_path: str):
        lines = []

        for result in results:
            lines.append(f"\n## {result['operator']} - {result['workload']}\n")
            lines.append(
                f"Shape: {result['workload_config'].get('shape', 'N/A')}, dtype: {result['workload_config'].get('dtype', 'float32')}\n"
            )
            lines.append("| Implementation | Time (ms) | Speedup vs Torch |\n")
            lines.append("|----------------|-----------|------------------|\n")

            for impl, r in result["results"].items():
                if "mean_ms" in r:
                    time_str = f"{r['mean_ms']:.3f}"
                    speedup_str = f"{r.get('speedup_vs_torch', 1.0):.2f}x"
                    lines.append(f"| {impl} | {time_str} | {speedup_str} |\n")

        with open(output_path, "w") as f:
            f.writelines(lines)
