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
from qwen_vl_utils import process_vision_info
from deepspeed.accelerator import get_accelerator
from safetensors import safe_open
import time
import requests
from PIL import Image
from transformers import AutoProcessor,  Qwen3VLForConditionalGeneration
from dictionary_learning import utils
SAES_ROOT = "saes"
ARCHITECTURE = "TopKTrainer"

layers_of_interest = [5, 10, 15, 20, 30, 35]

layer_SAEs = {}
for layer in layers_of_interest:
    trained_sae, _ = utils.load_dictionary(
        os.path.join(SAES_ROOT, f"qwen_activations_{layer}_{ARCHITECTURE}_wandb", "trainer_0"),
        device="cuda:0",
    )
    trained_sae.eval()
    layer_SAEs[layer] = trained_sae
features_of_interest= {
    "red":{"5": [20737, 24706, 39554, 39300, 31370, 34188, 51985, 3729, 36632, 60064, 35617, 21796, 4645, 28205, 44078, 13744, 49328, 40376, 57272, 19265, 49474, 20292, 61509, 4422, 13898, 54748, 13917, 734, 24697, 996, 51703, 36985], "10": [38535, 37129, 57359, 46736, 63635, 52503, 55835, 16545, 19492, 38056, 5688, 55481, 21691, 11457, 5190, 11259, 30546, 43480, 43353, 30686, 37598, 6112, 14182, 42732, 8430, 9973, 2427], "15": [8962, 34311, 32277, 24857, 37921, 35619, 17705, 63793, 23985, 4531, 16180, 18998, 38586, 38592, 34503, 65351, 56650, 7758, 22864, 59604, 63188, 12504, 56281, 42595, 17127, 34936, 52220, 18175], "20": [30467, 65032, 49680, 38418, 13078, 45591, 7576, 45849, 63398, 61991, 9258, 7084, 41008, 33078, 5944, 1464, 45389, 62288, 47186, 24419, 28645, 49894, 38120, 44776, 29930, 60782, 58351, 2929], "25": [22147, 40708, 2570, 56208, 28821, 27670, 53528, 6048, 7461, 49191, 8618, 15660, 20781, 29744, 2354, 60083, 50612, 17078, 37177, 23995, 6848, 56385, 1611, 21451, 14289, 33502, 39778, 21092, 53483, 2799, 23408, 10223, 49018, 12030], "30": [55041, 2946, 31753, 2579, 2709, 16533, 45975, 57239, 53279, 40486, 48681, 64043, 30145, 48453, 8903, 59978, 55883, 53079, 4191, 31201, 30698, 56442, 39803], "32": [51334, 35082, 61067, 15884, 47374, 8229, 63525, 21671, 26534, 49717, 4668, 40514, 46534, 14930, 42842, 5468, 608, 18026, 50416, 58867, 16378]},
    "blue": {"5": [39300, 26374, 17672, 10378, 51985, 8982, 18074, 31904, 4645, 48295, 46760, 4018, 35891, 63925, 37430, 14265, 23227, 51644, 45505, 46403, 20292, 56262, 8263, 13898, 33483, 44372, 51541, 42215, 31847, 20075, 19437, 42227, 59385, 18042], "10": [32642, 4356, 55172, 30471, 56331, 15760, 42004, 44702, 50080, 16545, 22306, 50081, 52390, 12076, 31660, 64050, 55481, 54465, 22852, 11259, 61262, 47694, 20816, 57813, 5846, 33495, 45016, 34395, 25439, 30559, 18785, 6626, 6112, 23527, 50031, 49909, 13304, 26491, 1404], "15": [42880, 8962, 34311, 59017, 4370, 24857, 2331, 34936, 43036, 20765, 17705, 63793, 4531, 16180, 38326, 38200, 49465, 16957, 29118, 38592, 21447, 21192, 21324, 7758, 22864, 59604, 55894, 12504, 56281, 38882, 42595, 17127, 42215, 11116, 53106, 58488, 6522, 9211, 50296], "20": [30467, 12036, 32525, 48401, 38418, 2707, 47251, 18833, 13078, 45591, 39321, 42275, 39843, 63398, 61991, 35755, 7084, 41008, 1464, 17082, 56897, 58820, 31815, 36686, 3280, 21342, 24420, 44776, 29930, 60782, 58351, 36340, 64895], "25": [12294, 33938, 16530, 1555, 7461, 58025, 49577, 15660, 38188, 38062, 57136, 3252, 44343, 56385, 33859, 54342, 57287, 51656, 38858, 12622, 62424, 861, 52317, 35295, 29929, 20970, 25323, 23020, 30828, 13806, 3312, 63219, 40443, 10367], "30": [22529, 31753, 25742, 39186, 45975, 4650, 9644, 1718, 63031, 24760, 63304, 11218, 4191, 1514, 30698, 35311, 23537, 45170, 56442], "32": [51334, 15878, 7817, 33418, 61067, 13, 55191, 64411, 32283, 24479, 20780, 27825, 4668, 27213, 59086, 49237, 37589, 42842, 5468, 61663, 8929, 26474, 58867, 15732]},
    "green": {"5": [20737, 34188, 34447, 51985, 46613, 18074, 21796, 4645, 20393, 44078, 64305, 62643, 33332, 59061, 16438, 20919, 58942, 24132, 20292, 64456, 13898, 49738, 6988, 21708, 30037, 42722, 16484, 35813, 27753, 52204, 45679, 14449, 32760, 24697], "10": [11649, 41346, 37507, 65164, 46093, 15760, 37777, 42004, 42647, 280, 36000, 42530, 44069, 302, 1970, 54198, 51512, 55481, 23610, 25275, 46268, 55229, 48317, 42181, 15306, 47694, 30931, 13780, 33495, 16343, 35930, 14177, 24161, 30563, 14948, 56165, 23142, 53094, 55787, 19565, 5998, 29423, 49400, 18430], "15": [42880, 8962, 34950, 34311, 4370, 24857, 34936, 17705, 15152, 63793, 4531, 16180, 38326, 49465, 7758, 22864, 59604, 63188, 12504, 56281, 17127, 60914, 58488, 50296], "20": [30467, 65032, 32654, 48401, 27025, 30612, 13078, 45591, 60952, 45849, 45721, 39321, 7576, 63398, 58792, 35755, 7084, 41008, 1464, 8901, 31815, 36686, 30802, 31961, 44776, 38120, 60782, 45424, 2929, 36340, 11637, 54646, 39674, 39422], "25": [16788, 278, 7461, 38310, 15660, 40375, 8504, 6848, 56385, 60228, 31557, 34760, 62024, 22987, 1997, 12622, 9679, 27992, 861, 6499, 21092, 20970, 50283, 60910, 3312, 371, 40443, 10367], "30": [51075, 10659, 40486, 31753, 38506, 12784, 45689, 45170, 35670, 47289, 62906, 9532], "32": [13058, 38408, 33418, 61067, 13, 18849, 58019, 20780, 62389, 57914, 2253, 59086, 46415, 58193, 14930, 49237, 41941, 5468, 52190, 34156, 58867]}
}
feature_indices = features_of_interest["red"]
alpha = 100
def make_hook(layer_id):
    def hook(module, input, output):
        original_dtype = output.dtype
        original_device = output.device

        sae = layer_SAEs[layer_id]

        try:
            feature_index = feature_indices[str(layer_id)]
        except:
            feature_index = feature_indices[int(layer_id)]

        if isinstance(feature_index, int):
            feature_index = [feature_index]
        
        encoded = sae.encode(output)

        x = encoded[:, :, [feature_index]]

        mean = x.mean()
        x = torch.where(x == 0, mean * alpha, x * alpha)

        encoded[:, :, [feature_index]] = x
        decoded = sae.decode(encoded)

        return decoded.to(device=original_device, dtype=original_dtype)

    return hook
    # def _hook(self, module, input, output):        
    #     self.feats.append(output.detach().float().cpu().tolist())

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
        self.feats = []
        self.layers_of_interest = layers_of_interest

        if (is_meta):
            # self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, dtype=torch.bfloat16, device_map = "cpu")
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)

        self.model.eval()
        print("\n\ndtype:",self.model.dtype,"\n\n")

        if self.dtype == torch.float16:
            self.model.half()
        elif self.dtype == torch.bfloat16:
            self.model.bfloat16()
        self.local_rank = int(os.environ.get("RANK", 0))
        
        # print("\n\n",os.environ.items(),"\n\n")
        # if self.local_rank==0:
        self.hook_handles = []
        for layer in self.layers_of_interest:
            print("registered hook for layer",layer)
            handle = self.model.model.language_model.layers[layer].register_forward_hook(
                make_hook(layer)
            )
            self.hook_handles.append(handle)
        


    def __call__(self,
                inputs=[{"text":"Describe this image:", "image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}],
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
                        inputs=[{"text":"Describe this image:", "image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}],
                        num_tokens=100,
                        do_sample=False, rank=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": inputs[0]["image"],
                    },
                    {"type": "text", "text": inputs[0]["text"]},
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
        # print("Keys in inputs:", inputs.keys())
        # if "pixel_values" in inputs:
        #     pv = inputs["pixel_values"]
        #     print(f"pixel_values shape={pv.shape}, dtype={pv.dtype}, sum={pv.sum().item():.4f}")
        # else:
        #     print("WARNING: pixel_values is MISSING from inputs")
        # if "pixel_values" in inputs:
        #     inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        #     print(f"pixel_values after cast: dtype={inputs['pixel_values'].dtype}, "
        #         f"device={inputs['pixel_values'].device}, "
        #         f"sum={inputs['pixel_values'].sum().item():.4f}")
        # print(f"image_grid_thw: {inputs['image_grid_thw']}, device={inputs['image_grid_thw'].device}")
        
        
        # self.model.to(self.device)
        # if rank == 1: 
        #     save_model_weight_stats(self.model, "qwen3_weight_stats_dist_rank1.json")
        start = time.time()
        generated_ids = self.model.generate(**inputs, max_new_tokens=num_tokens, do_sample=do_sample)
        gen_time = time.time()-start
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print("generated_ids_trimmed:", len(generated_ids_trimmed[0]))
        print("Tokens per second:",len(generated_ids_trimmed[0])/gen_time)
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
