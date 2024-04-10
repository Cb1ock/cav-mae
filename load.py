import torch
import timm
from src.models import CAVMAE

assert timm.__version__ == '0.4.5' # it is important to have right version of timm

model_path = 'egs/celebv-text/exp/testmae01-audioset-cav-mae-balNone-lr5e-5-epoch25-bs16-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75-a5/models/audio_model.25.pth'

# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024,
                     modality_specific_depth=11,
                     norm_pix_loss=True, tr_pos=False) # most models are trained with pixel normalization and non-trainabe positional embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print('miss=',miss, 'unexpected=', unexpected) # check if all weights are correctly loaded

# 定义随机输入
audio_input = torch.randn(1, 1024, 80) # 假设音频输入的形状为(批量大小, 音频长度, 音频特征维度)
visual_input = torch.randn(1, 3, 224, 224) # 假设视觉输入的形状为(批量大小, 颜色通道数, 图像高度, 图像宽度)

# 将输入移动到正确的设备上
audio_input = audio_input.to(device)
visual_input = visual_input.to(device)

# 前向传播
audio_embedding, visual_embedding = audio_model(audio_input, visual_input)

# 打印输出的形状
print("Audio embedding shape:", audio_embedding.shape)
print("Visual embedding shape:", visual_embedding.shape)