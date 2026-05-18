import os
import math
import time
import torch
import deepspeed
from utils import DSPipeline, Performance
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator import get_accelerator
from arguments import parser
import json
import psutil
from datetime import datetime
import threading
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
args = parser.parse_args()


os.environ["TP_SOCKET_IFNAME"]="eno1" 
os.environ["NCCL_SOCKET_IFNAME"]="eno1"
os.environ["GLOO_SOCKET_IFNAME"]="eno1"
os.environ["NCCL_DEBUG"]="INFO"

deepspeed.init_distributed(dist_backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if args.hf_baseline and world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)

if local_rank == 0:
    see_memory_usage("before init", True)

t0 = time.time()

pipe = DSPipeline(model_name=args.model,
                  dtype=data_type,
                  is_meta=True,
                  device=local_rank,
                  checkpoint_path=args.checkpoint_path,
                  trust_remote_code=args.trust_remote_code)

if local_rank == 0:
    print(f"initialization time: {(time.time()-t0) * 1000}ms")
    see_memory_usage("after init", True)

ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)

injection_policy = {
    Gemma3DecoderLayer: ("self_attn.o_proj", "mlp.down_proj"),
    SiglipEncoderLayer: ("self_attn.out_proj", "mlp.fc2")
}

def log_resource_utilization(model_name: str, output_dir: str = "."):
    safe_model_name = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{safe_model_name}_utilization_{timestamp}.json")

    cpu_percent = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    
    gpu_stats = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_stats.append({
                "device_id": i,
                "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(i) / (1024**3)
            })

    stats = {
        "timestamp": timestamp,
        "model_name": model_name,
        "system_ram": {
            "used_gb": vm.used / (1024**3),
            "total_gb": vm.total / (1024**3),
            "percent": vm.percent
        },
        "system_cpu": {
            "percent": cpu_percent
        },
        "gpu_vram": gpu_stats
    }

    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)
    

def print_weight_sample(model, label):
    w = model.model.language_model.layers[0].input_layernorm.weight
    print(f"[{label}] input_layernorm[0] sum={w.sum().item():.4f}, device={w.device}, dtype={w.dtype}")

print_weight_sample(pipe.model, "BEFORE deepspeed")

if "AWQ" in args.model:
    print("\n\nAWQ\n\n")
    visual_module = pipe.model.visual
    pipe.model.visual = torch.nn.Identity()

pipe.model = deepspeed.init_inference(
    pipe.model,
    tensor_parallel={"tp_size": world_size, "tp_grain_size": 8},
    dtype=data_type,
    replace_with_kernel_inject=False,
    injection_policy=injection_policy,
    max_out_tokens=args.max_tokens,
    save_mp_checkpoint_path=args.save_mp_checkpoint_path,
    **ds_kwargs
)
if "AWQ" in args.model:
    pipe.model.module.visual = visual_module.to(local_rank)

print_weight_sample(pipe.model.module, "AFTER deepspeed")


if local_rank == 0:
    see_memory_usage("after init_inference", True)

input_sentences = ["Describe this image:"]

if args.batch_size > len(input_sentences):
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

# iters = 30 if args.test_performance else 2
iters = 1
times = []
for i in range(iters):
    get_accelerator().synchronize()
    start = time.time()
    log_thread = threading.Thread(target=log_resource_utilization, args=(args.model,))
    log_thread.start()
    outputs = pipe(inputs,
            num_tokens=args.max_new_tokens,
            do_sample=(not args.greedy), rank = torch.distributed.get_rank())
    log_thread.join()
    
    get_accelerator().synchronize()
    end = time.time()
    times.append(end - start)

if local_rank == 0:
    print(f"generation time is {times[-1]} sec")
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        Performance.print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config, args.dtype, args.batch_size)
torch.distributed.destroy_process_group()