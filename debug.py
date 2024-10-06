import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_directory = "models/story"

model_name = "story"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)
outputs_dict = {}
inputs = tokenizer("hello, world", return_tensors="pt")

def hook_fn(layer_name):
    def hook(module, input, output):
        outputs_dict[layer_name] = {
            "input": input,
            "output": output
        }
    return hook
        
#for name, layer in model.named_modules():
    #print(f"layer name: {name}")


for name, param in model.named_parameters():
    print(name)

for name, layer in model.named_modules():
    layer_name = f"transformer_layer_{name}"
    layer.register_forward_hook(hook_fn(layer_name))
    
with torch.no_grad():
    model(**inputs)


x = outputs_dict['transformer_layer_lm_head']['output'][0]

#print(x - x_r > 100)
#print(x)

