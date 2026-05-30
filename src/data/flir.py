import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF

class FLIR(Dataset):
    # 🌟 核心改动 1：入参增加 name 和 train，符合官方 src/data/__init__.py 的getattr动态调用规范
    def __init__(self, args, name='FLIR', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.benchmark = benchmark  # 👈 强行补上这一句，默认为 False
        self.ext = ('.png', '.png') # 👈 顺便把官方可能还会检查的后缀元组也补上
        self.scale = args.scale[0]  # 从官方 args 自动读取放大倍数 (如 4)
        
        # 从官方 args 动态读取 patch_size，默认为 192
        self.patch_size_t = getattr(args, 'patch_size', 192)
        
        # 🌟 核心改动 2：根据 train 状态，动态切换训练集与验证集/测试集路径
        # 🔴 请在此处修正为你真实的硬盘 FLIR 绝对路径
        if self.train:
            self.rgb_dir = Path("C:/HKU/FLIR_ADAS_v2/images_rgb_train/data")
            self.thermal_dir = Path("C:/HKU/FLIR_ADAS_v2/images_thermal_train/data")
        else:
            self.rgb_dir = Path("C:/HKU/FLIR_ADAS_v2/images_rgb_val/data")
            self.thermal_dir = Path("C:/HKU/FLIR_ADAS_v2/images_thermal_val/data")
        
        # 过滤白名单
        extensions = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.JPG', '**/*.JPEG', '**/*.PNG']

        # 穿透扫描并排序 (保留你的优秀强对齐逻辑)
        sorted_thermal_paths = []
        for ext in extensions:
            sorted_thermal_paths.extend([p.resolve() for p in self.thermal_dir.rglob(ext)])
        self.thermal_paths = sorted(sorted_thermal_paths)

        sorted_rgb_paths = []
        for ext in extensions:
            sorted_rgb_paths.extend([p.resolve() for p in self.rgb_dir.rglob(ext)])
        self.rgb_paths = sorted(sorted_rgb_paths)
        
        # 时序双盲截断配对
        min_match_count = min(len(self.thermal_paths), len(self.rgb_paths))
        self.thermal_paths = self.thermal_paths[:min_match_count]
        self.rgb_paths = self.rgb_paths[:min_match_count]
        
        print(f"====== ⏳ FLIR 双模态官方 DataLoader 成功激活 ======")
        print(f" 模式: {'TRAIN 训练集' if self.train else 'VAL/TEST 测试集'} | 成功配对: {len(self.thermal_paths)} 组完美样本.")
        print(f"====================================================")
        
    def __len__(self):
        return len(self.thermal_paths)
        
    def __getitem__(self, idx):
        thermal_path = self.thermal_paths[idx]
        rgb_path = self.rgb_paths[idx]
        
        img_t_hr = Image.open(thermal_path).convert('L')
        img_rgb = Image.open(rgb_path).convert('RGB')
        
        w_t, h_t = img_t_hr.size  
        
        # 🌟 核心改动 3：增加测试集保障。训练时才用随机裁剪，测试/验证时必须固定中心裁剪！
        # 否则每次测试同一个图片切出来的局部都不一样，PSNR 指标会疯狂乱跳，无法写进论文。
        if self.train:
            x_t = random.randint(0, w_t - self.patch_size_t)
            y_t = random.randint(0, h_t - self.patch_size_t)
        else:
            x_t = (w_t - self.patch_size_t) // 2
            y_t = (h_t - self.patch_size_t) // 2
        
        # 空间同坐标等比例裁剪
        patch_t_hr = img_t_hr.crop((x_t, y_t, x_t + self.patch_size_t, y_t + self.patch_size_t))
        
        x_rgb, y_rgb = x_t * 2, y_t * 2
        patch_size_rgb = self.patch_size_t * 2
        patch_rgb = img_rgb.crop((x_rgb, y_rgb, x_rgb + patch_size_rgb, y_rgb + patch_size_rgb))
        
        # 下采样制作低清输入 (LR)
        size_lr = self.patch_size_t // self.scale
        patch_t_lr = patch_t_hr.resize((size_lr, size_lr), Image.BICUBIC)
        
        # TF.to_tensor returns 0..1; the EDSR pipeline expects 0..args.rgb_range.
        rgb_range = getattr(self.args, 'rgb_range', 255)
        tensor_t_lr = TF.to_tensor(patch_t_lr).mul(rgb_range)  # [1, 48, 48]
        tensor_rgb = TF.to_tensor(patch_rgb).mul(rgb_range)    # [3, 384, 384]
        tensor_t_hr = TF.to_tensor(patch_t_hr).mul(rgb_range)  # [1, 192, 192]
        
        # 🌟 核心改动 4：严格适配 EDSR 官方的返回值接口规范
        # 1. 两个输入端打包为元组 (tensor_t_lr, tensor_rgb) 传给双流模型
        # 2. 第二个元素是单通道的红外 GT (tensor_t_hr)
        # 3. 第三个元素是图片纯文件名字符串，供官方写日志和存图
        return (tensor_t_lr, tensor_rgb), tensor_t_hr, thermal_path.name

    # 🌟 核心改动 5：添加官方包装器（ConcatDataset）需要的必备打底方法，防止初始化闪退
    def set_scale(self, idx_scale):
        pass