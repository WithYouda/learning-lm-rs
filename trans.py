import torch
from safetensors.torch import load_file, save_file

# 加载safetensors模型
input_path = "/home/withyouda/2024-infiniTensor/exam-grading/learning-lm-rs/models/chat/model.safetensors"
output_path = "/home/withyouda/2024-infiniTensor/exam-grading/learning-lm-rs/models/half/chat/model.safetensors"
weights = load_file(input_path)

# 将权重转换为FP16
for key, value in weights.items():
    weights[key] = value.half()

# 保存为FP16的safetensors模型
save_file(weights, output_path)

print(f"模型已成功保存为FP16格式: {output_path}")
