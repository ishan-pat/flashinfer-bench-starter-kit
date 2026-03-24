import modal
from pathlib import Path
import sys

app = modal.App("flashinfer-bench-debug")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

project_dir = str(Path(__file__).parent.parent.absolute())

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "tomli")
    .add_local_dir(project_dir, remote_path="/project")
)

@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_benchmark_remote():
    import sys
    sys.path.insert(0, "/project")
    
    from scripts.pack_solution import pack_solution
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet, Solution
    
    print("Packing solution from source files remotely...")
    solution_path = pack_solution(Path("/project/solution.json"))
    
    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    
    config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
    
    import os
    trace_path = "/data/flashinfer-trace"
    if not os.path.exists(trace_path):
        trace_path = "/data"
        
    # === CLEAR PREVIOUS DEBUG LOGS ===
    if os.path.exists("/tmp/kernel_debug.txt"):
        os.remove("/tmp/kernel_debug.txt")
    print("Contents of trace dir:", os.listdir(trace_path))

    trace_set = TraceSet.from_path(trace_path)
    if solution.definition not in trace_set.definitions:
        print("Available definitions:", list(trace_set.definitions.keys()))
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
        
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")
        
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads[:1]},
        traces={definition.name: []},
    )
    
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=False)
    
    # === BYPASS STDOUT REDIRECTION: DUMP KERNEL LOGS ===
    log_text = ""
    if os.path.exists("/tmp/kernel_debug.txt"):
        with open("/tmp/kernel_debug.txt", "r") as f:
            log_text = f.read()
        print("\n=== KERNEL DEBUG LOGS ===")
        print(log_text)
    else:
        print("No debug logs found at /tmp/kernel_debug.txt")
    print("=========================\n")
    
    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}
    
    for trace in traces:
        if getattr(trace, "evaluation", None):
            entry = {
                "status": trace.evaluation.status.value,
            }
            if getattr(trace.evaluation, "performance", None):
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if getattr(trace.evaluation, "correctness", None):
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["raw_eval"] = trace.evaluation.model_dump_json() if hasattr(trace.evaluation, "model_dump_json") else str(trace.evaluation)
            results[definition.name][trace.workload.uuid] = entry
            
    print("\n=== BENCHMARK RESULTS ===")
    for def_name, def_traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in def_traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")
            if "latency_ms" in result:
                print(f" | {result['latency_ms']:.3f} ms | {result['speedup_factor']:.2f}x speedup", end="")
            if "max_abs_error" in result:
                abs_err = result["max_abs_error"]
                print(f" | abs_err={abs_err:.2e}", end="")
            if "raw_eval" in result:
                print(f" | RAW: {result['raw_eval']}", end="")
            print()
    print("=========================\n")
            
    return results, log_text
    
@app.local_entrypoint()
def main():
    results, log_text = run_benchmark_remote.remote()
    
    # Save logs locally with a timestamp
    import datetime
    from pathlib import Path
    
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/logs_debug_{timestamp}.txt"
    
    with open(log_file, "w") as f:
        f.write(log_text)
    print(f"Captured remote logs saved to: {log_file}")
