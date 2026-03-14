import os
import math
import time
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils import DSPipeline, Performance
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator import get_accelerator
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    # Qwen3VLVisionBlock,
    Qwen3VLTextDecoderLayer,
)
from arguments import parser

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
    Qwen3VLTextDecoderLayer: ("self_attn.o_proj", "mlp.down_proj"),
    # Qwen3VLVisionBlock: ("attn.proj", "mlp.linear_fc2"),
}

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

if local_rank == 0:
    see_memory_usage("after init_inference", True)

input_sentences = ["Describe this image:"]

if args.batch_size > len(input_sentences):
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]


iters = 30 if args.test_performance else 2
times = []

for i in range(iters):
    get_accelerator().synchronize()
    start = time.time()
    outputs = pipe(inputs,
            num_tokens=args.max_new_tokens,
            do_sample=(not args.greedy))
    get_accelerator().synchronize()
    end = time.time()
    times.append(end - start)

if local_rank == 0:
    print(f"generation time is {times[1]} sec")
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        Performance.print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config, args.dtype, args.batch_size)