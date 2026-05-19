from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration
import json
from PIL import Image
import requests
import torch
max_memory_mapping = {0: "13GB", "cpu":"29GB"}

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto"#, max_memory = max_memory_mapping
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
print(model.model.language_model.layers)

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

# save_model_weight_stats(model, "qwen3_weight_stats.json")


# model_id = "google/gemma-3-12b-it"

# model = Gemma3ForConditionalGeneration.from_pretrained(
#     model_id, device_map="auto", max_memory = max_memory_mapping
# ).eval()

# print(model)

# processor = AutoProcessor.from_pretrained(model_id)

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
#             {"type": "text", "text": "Describe this image in detail."}
#         ]
#     }
# ]

# inputs = processor.apply_chat_template(
#     messages, add_generation_prompt=True, tokenize=True,
#     return_dict=True, return_tensors="pt"
# ).to(model.device, dtype=torch.bfloat16)

# input_len = inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]

# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)
