import gc
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
model_path = root / 'source/CENet/CENET/resnet34_traced_forC.pt'
torch.manual_seed(0)
report = {'purpose': 'synthetic input deployment smoke test, not segmentation accuracy or end-to-end latency',
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
          'gpu': torch.cuda.get_device_name(0), 'shape': [1, 5, 64, 2048],
          'commit': subprocess.check_output(['git', '-C', str(root/'source'), 'rev-parse', 'HEAD'], text=True).strip(),
          'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'measurements': []}

def tensors(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        return [t for item in output for t in tensors(item)]
    if isinstance(output, dict):
        return [t for item in output.values() for t in tensors(item)]
    raise TypeError(type(output))

for name, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
    model = torch.jit.load(str(model_path), map_location='cuda').eval().to(dtype=dtype)
    x = torch.randn(report['shape'], device='cuda', dtype=dtype)
    with torch.inference_mode():
        for _ in range(10):
            output = model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
        values = tensors(output)
        assert values and all(torch.isfinite(t).all().item() for t in values), 'Non-finite output'
        result = {'precision': name, 'warmup': 10, 'samples': 50,
                  'median_ms': statistics.median(elapsed), 'mean_ms': statistics.mean(elapsed),
                  'p95_ms': sorted(elapsed)[47], 'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
                  'outputs': [list(t.shape) for t in values], 'finite': True}
        report['measurements'].append(result)
        print(json.dumps(result), flush=True)
    del model, x, output, values
    gc.collect()
    torch.cuda.empty_cache()
(root/'results').mkdir(exist_ok=True)
(root/'results/smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
