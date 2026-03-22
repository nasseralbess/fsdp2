import torch
from model import Transformer
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard
import os
import io
from pathlib import Path
import json
import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, Qwen3VLForConditionalGeneration, AutoProcessor
from deepspeed.accelerator import get_accelerator
from safetensors import safe_open

def inspect_model(model: FSDPModule):
    # assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    # if torch.distributed.get_rank() == 0:
    print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        # assert param.dtype == torch.float32
        # print(param.get_local_tensor())


def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()


'''
Helper classes and functions for DeepSpeed
'''



class DSPipeline():
    def __init__(self,
                 model_name='Qwen/Qwen3-VL-8B-Thinking',
                 dtype=torch.bfloat16,
                 is_meta=True,
                 device="cuda:0",
                 checkpoint_path=None,
                 trust_remote_code=True,
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(get_accelerator().device_name(device))

        self.processor = AutoProcessor.from_pretrained(self.model_name, dtype=self.dtype)
        self.processor.tokenizer.padding_side = 'left'

        if (is_meta):
            # self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)

            # with deepspeed.OnDevice(dtype=self.dtype, device="meta"):
            #     self.model = Qwen3VLForConditionalGeneration._from_config(self.config)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, dtype=torch.bfloat16, device_map = "cpu")
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)

        self.model.eval()
        print("\n\ndtype:",self.model.dtype,"\n\n")

        if self.dtype == torch.float16:
            self.model.half()
        elif self.dtype == torch.bfloat16:
            self.model.bfloat16()
        self.model.model.language_model.layers[0].register_forward_hook(self._hook)
        print("\n\ndtype2:",self.model.dtype,"\n\n")


    def _hook(self, module, input, output):
        feats = {}
        feats['feat'] = output.detach().float().cpu().tolist()
        json.dump(feats, open("activation.json", "w"))

    def __call__(self,
                 inputs=["test"],
                 num_tokens=100,
                 do_sample=False, rank=None):
        if isinstance(inputs, str):
            inputs = [inputs]

        outputs = self.generate_outputs(inputs, num_tokens=num_tokens, do_sample=do_sample, rank=rank)
        return outputs

    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(self.model_name,
                                          allow_patterns=["*"],
                                          cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                          local_files_only=False,
                                          revision=None)
        else:
            assert os.path.exists(checkpoint_path)
            repo_root = checkpoint_path

        checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")

        with io.open(checkpoints_json, "w", encoding="utf-8") as f:
            file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.safetensors") if entry.is_file()]
            if not file_list:
                file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
            
            data = {"type": "DS_MODEL", "checkpoints": file_list, "version": 1.0}
            json.dump(data, f)

        return repo_root, checkpoints_json

    def generate_outputs(self,
                         inputs=["Describe this image:"],
                         num_tokens=100,
                         do_sample=False, rank=None):
        # print("generate_outputs called")
        # conversations = [
        #     [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "image",
        #                     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        #                 },
        #                 {"type": "text", "text": text_input},
        #             ],
        #         }
        #     ]
        #     for text_input in inputs
        # ]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "image": "demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]


        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        print("Keys in inputs:", inputs.keys())
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            print(f"pixel_values shape={pv.shape}, dtype={pv.dtype}, sum={pv.sum().item():.4f}")
        else:
            print("WARNING: pixel_values is MISSING from inputs")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
            print(f"pixel_values after cast: dtype={inputs['pixel_values'].dtype}, "
                f"device={inputs['pixel_values'].device}, "
                f"sum={inputs['pixel_values'].sum().item():.4f}")
        print(f"image_grid_thw: {inputs['image_grid_thw']}, device={inputs['image_grid_thw'].device}")
        
        
        # self.model.to(self.device)
        if rank == 1: 
            save_model_weight_stats(self.model, "qwen3_weight_stats_dist_rank1.json")
        generated_ids = self.model.generate(**inputs, max_new_tokens=num_tokens, do_sample=do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # print("generted_ids:")
        # for i in generated_ids[0].detach().cpu():
        #     print(i, end=", ")
        # print("\n")
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
def save_model_weight_stats(model, output_path="weight_stats.json"):
    stats_dict = {}
    
    for name, param in model.named_parameters():
        if param.is_meta:
            if '.' in name:
                module_name, param_name = name.rsplit('.', 1)
                module = model.get_submodule(module_name)
            else:
                module = model
                param_name = name
                
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "weights_map"):
                param_data = module._hf_hook.weights_map[param_name].float()
            else:
                continue
        else:
            param_data = param.data.float()
        
        keys = name.split('.')
        current_level = stats_dict
        
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
            
        current_level[keys[-1]] = {
            "sum": param_data.sum().item(),
            "mean": param_data.mean().item()
        }
        
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)

class Performance():

    def print_perf_stats(latency_set, config, dtype, batch_size, warmup=3):
        # trim warmup queries
        latency_set = list(latency_set)
        latency_set = latency_set[warmup:]
        count = len(latency_set)

        if count > 0:
            latency_set.sort()
            avg = sum(latency_set) / count
            num_layers = getattr(config, "num_layers", config.num_hidden_layers)
            num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
            if dtype == "float16":
                num_bytes = 2
            elif dtype == "float32":
                num_bytes = 4
            else:
                num_bytes = 1
            print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
            print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
            print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * batch_size / 1e12))
