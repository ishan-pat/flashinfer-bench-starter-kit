"""Direct kernel test on B200 — uses REAL benchmark data to find the segfault!"""
import modal
from pathlib import Path

app = modal.App("flashinfer-diag")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
project_dir = str(Path(__file__).parent.parent.absolute())

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench==0.1.2", "torch", "triton", "numpy", "tomli")
    .add_local_dir(project_dir, remote_path="/project")
)

@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def diagnose():
    import sys, traceback
    sys.path.insert(0, "/project")
    import torch
    print(f"torch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0)}")

    from solution.triton.kernel import kernel
    from flashinfer_bench import TraceSet, Solution
    from scripts.pack_solution import pack_solution
    from pathlib import Path

    solution_path = pack_solution(Path("/project/solution.json"))
    solution = Solution.model_validate_json(solution_path.read_text())

    trace_set = TraceSet.from_path("/data/flashinfer-trace")
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        return "No workloads found."

    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    
    wl = workloads[0]
    print(f"\nWorkload loaded: {getattr(wl, 'name', 'unknown-workload')}")

    device = "cuda:0"
    trace_root = Path("/data/flashinfer-trace")
    
    actual_workload = wl.workload if hasattr(wl, "workload") else wl
    trace = wl.traces[0] if hasattr(wl, "traces") else wl
    if hasattr(trace, "workload"): actual_workload = trace.workload
    
    print(f"Loading inputs for workload...")
    safe_tensors = load_safetensors(definition, actual_workload, trace_root)
    inputs = gen_inputs(definition, actual_workload, device, safe_tensors=safe_tensors)
    
    for i, t in enumerate(inputs):
        if isinstance(t, torch.Tensor):
            print(f"  inp[{i}]: shape={list(t.shape)} dtype={t.dtype} strides={t.stride()}")
        else:
            print(f"  inp[{i}]: {t}")

    print(f"Allocating outputs...")
    out = allocate_outputs(definition, inputs, device)
    print(f"  out[0]: shape={list(out[0].shape)} dtype={out[0].dtype}")

    print("Running kernel wrapper on true workload...")
    try:
        kernel_args = list(inputs) + list(out)
        kernel(*kernel_args)
        torch.cuda.synchronize()
        print("Kernel completed successfully!")
        
        print(f"safetensors keys: {list(safe_tensors.keys())}")
        out_key = None
        # Try specific known output keys or any key that isn't input-like
        candidates = [k for k in safe_tensors.keys() if "out" in k.lower() or "final" in k.lower() or "output" in k.lower()]
        if not candidates:
            # Fallback: find the tensor that matches our output shape [seq_len, 7168]
            expected_shape = list(out[0].shape)
            for k, v in safe_tensors.items():
                if list(v.shape) == expected_shape:
                    out_key = k
                    break
        else:
            out_key = candidates[0]
        
        if out_key:
            print(f"Comparing against reference (key: {out_key})...")
            ref_out = [safe_tensors[out_key].to(device)]
            for i, (o, ref) in enumerate(zip(out, ref_out)):
                diff = torch.abs(o - ref)
                abs_err = torch.max(diff).item()
                print(f"Output {i} max abs error: {abs_err}")
                if abs_err > 0.1:
                    max_idx = torch.argmax(diff)
                    max_coord = tuple(c.item() for c in torch.unravel_index(max_idx, diff.shape))
                    print(f"  Max error at {max_coord}: our value={o[max_coord].item()}, ref value={ref[max_coord].item()}")
                    
                    # Check 5 random mismatched elements to see the pattern
                    mismatches = diff > 0.1
                    mismatch_indices = mismatches.nonzero()[:5]
                    print(f"  Sample mismatches:")
                    for idx in mismatch_indices:
                        coord = tuple(c.item() for c in idx)
                        print(f"    {coord}: our={o[coord].item()}, ref={ref[coord].item()}")
        else:
            print("Could not find output reference in safetensors!")
                
    except Exception as e:
        print(f"\nKERNEL ERROR:")
        traceback.print_exc()
        return f"ERROR: {e}"

    return "SUCCESS"

@app.local_entrypoint()
def main():
    result = diagnose.remote()
    print(f"\n=== RESULT ===\n{result}")
