import torch, inspect
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

MODEL_DIR = "results/GDSC2/baseline_models/exclude_target"  # 改成你的基线模型保存目录
m = SimpleNeuralNetwork.load(MODEL_DIR)

# 1) 打印整体结构
print(m.model)  # 直接肉眼看有没有 Dropout(...)

# 2) 统计所有 Dropout 层及其 p
dropouts = [(name, mod.p) for name, mod in m.model.named_modules()
            if isinstance(mod, torch.nn.Dropout)]
print(f"[CHECK] #Dropout layers = {len(dropouts)}")
for name, p in dropouts:
    print(f"  - {name}: p={p}")
