import argparse
import os
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import torch
import torch.optim as optim
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_mixed_precision, inspect_model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from accelerate import init_empty_weights
from transformers.modeling_utils import load_state_dict
from transformers.utils import cached_file
from safetensors.torch import load_file
from torch.distributed.tensor import distribute_tensor


from khaled import UNet, Trainer, PairedImageDataset

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

    
def load_train_objs():
    train_set = PairedImageDataset(face_paths=face_train, comic_paths=comics_train, transform=transform)  
    model = UNet()
    
    return train_set, model

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )



def main(args):
    os.environ["TP_SOCKET_IFNAME"]="lo0" 
    os.environ["NCCL_SOCKET_IFNAME"]="lo0"
    os.environ["GLOO_SOCKET_IFNAME"]="lo0"
    os.environ["NCCL_DEBUG"]="INFO"
    _min_gpu_count = 1
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    # rank = int(os.environ["LOCAL_RANK"])
    rank=0
    # if torch.accelerator.is_available():
    #     # device_type = torch.accelerator.current_accelerator()
        # device = torch.device(f"{device_type}:{rank}")
        # torch.accelerator.device_index(rank)
        # print(f"Running on rank {rank} on device {device}")
    # else:
    device = torch.device("cpu")
    print(f"Running on device {device}")

    # backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend="gloo",  init_method='tcp://127.0.0.1:12355', rank=0, world_size=1)
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    torch.manual_seed(0)

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        
    mesh = init_device_mesh("cpu", mesh_shape=(1,))
    
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, **fsdp_kwargs)
    
    fully_shard(model.lm_head, mesh=mesh, **fsdp_kwargs)
    fully_shard(model.model, mesh=mesh, **fsdp_kwargs)
    fully_shard(model, mesh=mesh, **fsdp_kwargs)

    model.to_empty(device=device)

    from safetensors import safe_open

    archive_file = cached_file("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors")
    named_params = dict(model.named_parameters())
    with safe_open(archive_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key not in named_params:
                continue
            tensor = f.get_tensor(key)
            param = named_params[key]
            dtensor = distribute_tensor(tensor, param.device_mesh, param.placements)
            state_dict_chunk = {key: dtensor}
            
            if key == "model.embed_tokens.weight" and "lm_head.weight" in named_params:
                lm_head_param = named_params["lm_head.weight"]
                state_dict_chunk["lm_head.weight"] = distribute_tensor(
                    tensor, lm_head_param.device_mesh, lm_head_param.placements
                )
                
            model.load_state_dict(state_dict_chunk, strict=False)
                
            model.load_state_dict(state_dict_chunk, strict=False)
            del tensor
    
    
    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
    print("before checkpointer")
   

    

    inspect_model(model)
    from torch.distributed.tensor import DTensor

    local_params = 0
    for p in model.parameters():
        if isinstance(p, DTensor):
            local_params += p._local_tensor.numel()
        else:
            local_params += p.numel()

    print(
        f"[rank {torch.distributed.get_rank()}] "
        f"local shard params = {local_params}"
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    if args.mixed_precision:
        inspect_mixed_precision(model)
    # if checkpointer.last_training_time is not None:
    #     checkpointer.load_optim(model, optimizer)
    print("after optim load")
    print("in inference")
    if args.explicit_prefetching:
        model.unshard()
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model(**model_inputs)
    print(out)
    # print(tokenizer.decode(out[0], skip_special_tokens=True))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 16)')
    args = parser.parse_args()
    
    main(args)
