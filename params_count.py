# videos: torch.Size([256, 3, 16, 224, 224])
from thop import profile
from thop import clever_format
import argparse
import torch
import os
import sys
sys.path.append("/home/smartgrasping/rcao/Surgformer")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from timm.models import create_model

from model.surgformer_base import surgformer_base
from model.surgformer_HTA import surgformer_HTA
from model.surgformer_HTA_KCA import surgformer_HTA_KCA

model = create_model(
    'surgformer_base',
    pretrained=True,
    pretrain_path='pretrain_weights/TimeSformer_divST_8x32_224_K400.pyth',
    num_classes=7,
    all_frames=16,
    fc_drop_rate=0.5,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    drop_block_rate=None,
)

# 121.106M 24.323T
device = torch.device("cuda:0")
model = model.to(device)
model = model.eval()
input = torch.randn(42, 3, 16, 224, 224).to(device)
with torch.no_grad():
    flops, params = profile(model, (input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(params)
    print(flops)
    