# DDPM Colab 抄写练习计划

## 目标

通过在 Colab 中逐步抄写 `denoising-diffusion-pytorch` 的核心代码，深入理解 DDPM 的原理和实现。

整个练习聚焦于**核心文件** `denoising_diffusion_pytorch.py`（~1700行）和辅助文件 `attend.py`（~215行），按模块由浅入深拆分为 **6 个 Session**。

---

## 总览：代码架构

```
denoising_diffusion_pytorch/
├── attend.py                  ← Session 1（Attention 工具模块）
├── denoising_diffusion_pytorch.py  ← Session 2-6（核心实现）
│   ├── helpers & utils        ← Session 2
│   ├── building blocks        ← Session 3（Block, ResnetBlock, Attention）
│   ├── Unet                   ← Session 4（U-Net 编码器-解码器）
│   ├── GaussianDiffusion      ← Session 5（扩散过程）
│   └── Trainer                ← Session 6（训练循环）
└── (其他变体文件 - 可选进阶)
```

---

## Session 1: Attend 模块 (~215 行)

**文件**: `attend.py`
**预计时间**: 30-45 分钟
**核心概念**: Scaled Dot-Product Attention, Flash Attention

### 抄写内容

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 1-24 | imports + `AttentionConfig` namedtuple | `SDPBackend` 枚举 |
| 26-74 | `exists()`, `default()`, `once()` 工具函数 | 装饰器模式、闭包 |
| 76-132 | `Attend.__init__()` | Flash Attention 硬件检测，GPU compute capability |
| 134-171 | `Attend.flash_attn()` | `F.scaled_dot_product_attention`，自定义 scale |
| 173-215 | `Attend.forward()` | 标准 attention: `softmax(QK^T/√d)V`，`einsum` 用法 |

### 练习要点
- [ ] 理解 `einsum("b h i d, b h j d -> b h i j", q, k)` 的含义
- [ ] 对比标准 attention 和 Flash Attention 的区别
- [ ] 思考：为什么需要根据 GPU 型号选择不同的 backend？

### 验证
```python
# 在 Colab 中测试
attend = Attend(flash=False)
q = torch.randn(2, 4, 16, 32)  # batch=2, heads=4, seq=16, dim=32
k = torch.randn(2, 4, 16, 32)
v = torch.randn(2, 4, 16, 32)
out = attend(q, k, v)
print(out.shape)  # torch.Size([2, 4, 16, 32])
```

---

## Session 2: 工具函数 & 基础模块 (~150 行)

**文件**: `denoising_diffusion_pytorch.py` 第 34-178 行
**预计时间**: 30-45 分钟
**核心概念**: 工具函数、图像归一化、上下采样

### 抄写内容

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 34-67 | imports | `einops`, `ema_pytorch`, `accelerate` 生态 |
| 68-73 | `ModelPrediction` namedtuple | 存储模型预测结果 |
| 77-137 | 工具函数 | `exists`, `default`, `cast_tuple`, `cycle`, `num_to_groups` |
| 139-153 | 归一化函数 | `[0,1] ↔ [-1,1]` 转换，为什么扩散模型用 `[-1,1]` |
| 155-178 | `Upsample` & `Downsample` | `nn.Upsample` + Conv vs. PixelShuffle 逆操作 |

### 练习要点
- [ ] 理解 `Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)` 的 space-to-depth 操作
- [ ] 对比 Downsample（PixelUnshuffle + 1x1 Conv）vs 普通 stride=2 Conv 的优劣
- [ ] `cycle(dl)` 无限循环 DataLoader 的用途

### 验证
```python
down = Downsample(64)
up = Upsample(64)
x = torch.randn(1, 64, 32, 32)
print(down(x).shape)  # [1, 64, 16, 16]
print(up(down(x)).shape)  # [1, 64, 32, 32]
```

---

## Session 3: 构建块 — Norm, Embedding, Block, Attention (~250 行)

**文件**: `denoising_diffusion_pytorch.py` 第 180-428 行
**预计时间**: 60-90 分钟
**核心概念**: RMSNorm, 正弦位置编码, ResNet Block, Linear/Full Attention

### 抄写内容

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 180-193 | `RMSNorm` | 与 LayerNorm 的区别，`F.normalize` |
| 197-248 | `SinusoidalPosEmb` + `RandomOrLearnedSinusoidalPosEmb` | 时间步编码，傅里叶特征 |
| 252-316 | `Block` + `ResnetBlock` | 自适应归一化 (scale_shift), 时间条件化, 残差连接 |
| 318-377 | `LinearAttention` | O(n) 线性注意力，`softmax` 分离技巧，memory KV |
| 379-428 | `Attention` | O(n²) 标准注意力，memory KV tokens |

### 练习要点
- [ ] **SinusoidalPosEmb**: 画出不同频率的 sin/cos 曲线，理解为什么能编码时间步
- [ ] **ResnetBlock 中的 time conditioning**: `time_emb → MLP → chunk(2) → scale, shift → x * (scale+1) + shift`
- [ ] **LinearAttention vs Attention**:
  - Linear: `Q(K^TV)` — 先算 `K^TV`，复杂度 O(n)
  - Full: `(QK^T)V` — 先算 `QK^T`，复杂度 O(n²)
- [ ] **Memory KV**: 可学习的全局上下文 token，类似 CLS token

### 验证
```python
# 测试时间步编码
emb = SinusoidalPosEmb(128)
t = torch.tensor([0, 100, 500, 999])
print(emb(t).shape)  # [4, 128]

# 测试 ResnetBlock
block = ResnetBlock(64, 128, time_emb_dim=256)
x = torch.randn(2, 64, 32, 32)
t_emb = torch.randn(2, 256)
print(block(x, t_emb).shape)  # [2, 128, 32, 32]

# 测试 LinearAttention vs Attention
la = LinearAttention(64)
fa = Attention(64)
x = torch.randn(2, 64, 16, 16)
print(la(x).shape)  # [2, 64, 16, 16]
print(fa(x).shape)  # [2, 64, 16, 16]
```

---

## Session 4: U-Net 架构 (~240 行)

**文件**: `denoising_diffusion_pytorch.py` 第 432-667 行
**预计时间**: 60-90 分钟
**核心概念**: U-Net 编码器-解码器, Skip Connection, Time Conditioning

### 抄写内容

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 432-501 | `Unet.__init__` — 参数定义 & 维度计算 | `dim_mults` 如何控制每层通道数 |
| 513-533 | 时间嵌入 MLP | `SinusoidalEmb → Linear → GELU → Linear` |
| 535-572 | 编码器 (downs) | 每层: `ResBlock → ResBlock → Attn → Downsample` |
| 574-593 | 瓶颈 (mid) + 解码器 (ups) | 瓶颈用 Full Attention; 解码器拼接 skip connection |
| 595-667 | 输出层 + `forward()` | encoder→bottleneck→decoder 的完整数据流 |

### 练习要点
- [ ] **画出 U-Net 结构图**: 以 `dim_mults=(1,2,4,8)`, `dim=64` 为例
  ```
  输入: [B, 3, 64, 64]
  ↓ init_conv: [B, 64, 64, 64]
  ↓ Encoder L1: [B, 64, 64, 64] → ↓ [B, 128, 32, 32]  (存2个skip)
  ↓ Encoder L2: [B, 128, 32, 32] → ↓ [B, 256, 16, 16]  (存2个skip)
  ↓ Encoder L3: [B, 256, 16, 16] → ↓ [B, 512, 8, 8]    (存2个skip)
  ↓ Encoder L4: [B, 512, 8, 8] → [B, 512, 8, 8]         (存2个skip, 无downsample)
  ↓ Bottleneck: [B, 512, 8, 8] (ResBlock + FullAttn + ResBlock)
  ↑ Decoder L4: cat skip → [B, 512, 8, 8]
  ↑ Decoder L3: cat skip → ↑ [B, 256, 16, 16]
  ↑ Decoder L2: cat skip → ↑ [B, 128, 32, 32]
  ↑ Decoder L1: cat skip → ↑ [B, 64, 64, 64]
  → cat with r → final_res_block → final_conv: [B, 3, 64, 64]
  ```
- [ ] 理解 skip connection 的 `h.append()` 和 `h.pop()` 配对（LIFO 栈结构）
- [ ] `self_condition`: 如何将上一次预测 concat 到输入

### 验证
```python
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=3)
x = torch.randn(2, 3, 64, 64)
t = torch.randint(0, 1000, (2,))
print(model(x, t).shape)  # [2, 3, 64, 64]

# 打印参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~34M
```

---

## Session 5: GaussianDiffusion — 扩散过程 (~590 行)

**文件**: `denoising_diffusion_pytorch.py` 第 669-1331 行
**预计时间**: 90-120 分钟（最核心的部分）
**核心概念**: 前向扩散, 反向去噪, DDPM/DDIM 采样, 损失函数

### 5a: Beta Schedule & 初始化 (第 669-903 行)

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 671-688 | `extract()` 函数 | 从预计算数组中按 timestep 索引 |
| 690-736 | 三种 beta schedule | linear / cosine / sigmoid 的区别和适用场景 |
| 738-903 | `GaussianDiffusion.__init__()` | **重点**: 所有预计算变量的数学含义 |

**关键数学**:
```
β_t          : 每步加噪方差
α_t = 1 - β_t : 每步信号保留率
ᾱ_t = ∏α_i   : 累积信号保留率 (alphas_cumprod)

前向过程: q(x_t|x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)·I)
后验:     q(x_{t-1}|x_t,x_0) = N(μ̃_t, β̃_t)
  其中  μ̃_t = (β_t·√ᾱ_{t-1})/(1-ᾱ_t)·x_0 + ((1-ᾱ_{t-1})·√α_t)/(1-ᾱ_t)·x_t
        β̃_t = β_t·(1-ᾱ_{t-1})/(1-ᾱ_t)
SNR_t = ᾱ_t / (1-ᾱ_t)
```

### 5b: 预测转换函数 (第 905-1010 行)

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 909-948 | `predict_start_from_noise/v` | noise ↔ x_0 ↔ v 三种表示的相互转换 |
| 950-969 | `q_posterior()` | 后验分布 q(x_{t-1}\|x_t, x_0) 的均值和方差 |
| 971-1010 | `model_predictions()` | 统一三种 objective 的预测接口 |

### 5c: 采样过程 (第 1012-1198 行)

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 1037-1091 | `p_sample()` + `p_sample_loop()` | DDPM 采样: 从 T 到 0 逐步去噪 |
| 1093-1161 | `ddim_sample()` + `sample()` | DDIM: 更少步数的确定性采样 |
| 1163-1198 | `interpolate()` | 潜空间插值 |

### 5d: 训练 (第 1200-1331 行)

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 1214-1241 | `q_sample()` | 前向扩散: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε |
| 1243-1311 | `p_losses()` | 训练损失: self-conditioning, offset noise, MSE + SNR weighting |
| 1313-1331 | `forward()` | 随机采样 timestep + 计算 loss |

### 练习要点
- [ ] **画出 beta schedule 曲线**:
  ```python
  import matplotlib.pyplot as plt
  t = 1000
  plt.plot(linear_beta_schedule(t), label='linear')
  plt.plot(cosine_beta_schedule(t), label='cosine')
  plt.plot(sigmoid_beta_schedule(t), label='sigmoid')
  plt.legend(); plt.show()
  ```
- [ ] **画出 alpha_cumprod 和 SNR 曲线** — 直观理解不同时间步的噪声程度
- [ ] **对比 DDPM vs DDIM 采样**:
  - DDPM: 1000 步，随机性，`x_{t-1} = μ + σ·z`
  - DDIM: 可以 50 步，eta=0 时确定性，eta=1 时等效 DDPM
- [ ] **理解 min-SNR loss weighting**: 为什么需要？高噪声时间步的 loss 太大会主导训练

### 验证
```python
model = Unet(dim=64, dim_mults=(1, 2, 4), channels=3)
diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000, sampling_timesteps=50)

# 前向扩散可视化
img = torch.randn(1, 3, 32, 32) * 0.5  # 模拟干净图像
for t_val in [0, 250, 500, 750, 999]:
    t = torch.tensor([t_val])
    noisy = diffusion.q_sample(img, t)
    print(f"t={t_val}: noise_level={noisy.std():.3f}")

# 训练一步
loss = diffusion(torch.randn(4, 3, 32, 32))
print(f"Loss: {loss.item():.4f}")

# 采样（需要 GPU，耗时较长）
# samples = diffusion.sample(batch_size=4)
```

---

## Session 6: Trainer & Dataset (~380 行)

**文件**: `denoising_diffusion_pytorch.py` 第 1332-1714 行
**预计时间**: 45-60 分钟
**核心概念**: 训练循环, EMA, Checkpoint, FID 评估

### 抄写内容

| 行范围 | 内容 | 关键知识点 |
|--------|------|-----------|
| 1334-1379 | `Dataset` 类 | PIL 图像加载, `transforms`, 归一化 |
| 1383-1500 | `Trainer.__init__()` | `Accelerator`, EMA, 梯度累积, 混合精度 |
| 1500-1600 | `Trainer.train()` 核心循环 | 训练步骤: load → forward → loss → backward → step |
| 1600-1714 | Checkpoint & 采样 | 保存/加载, 定期生成样本, FID 评估 |

### 练习要点
- [ ] **EMA (Exponential Moving Average)**: 为什么用 EMA 的权重来做推理而非训练权重？
- [ ] **梯度累积**: `gradient_accumulate_every` 如何实现 effective batch size 放大
- [ ] **`Accelerator`**: 如何自动处理多 GPU / 混合精度
- [ ] **FID**: 计算流程（Inception-v3 特征提取 → 均值和协方差 → Fréchet 距离）

### 验证（完整训练 demo）
```python
# Colab 中的完整训练示例
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=250)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size=16,
    train_lr=8e-5,
    train_num_steps=10000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,
    save_and_sample_every=1000,
)
trainer.train()
```

---

## 可选进阶 Sessions

完成核心 6 个 Session 后，可以选择以下模块深入：

### Session 7: Elucidated Diffusion (~670 行)
- Karras et al. 设计空间
- Preconditioning (c_skip, c_out, c_in, c_noise)
- Heun's 二阶 ODE 求解器
- 随机采样 (churn)

### Session 8: Classifier-Free Guidance (~1780 行)
- 条件/无条件联合训练
- Guidance scale 的作用
- 文本引导图像生成基础

### Session 9: 1D Diffusion (~1789 行)
- 将扩散模型应用于序列数据
- 1D 卷积 vs 2D 卷积的区别
- 时间序列 / 音频生成

---

## Colab 环境配置

每个 Session 开头都需要的设置：

```python
# Cell 1: 安装依赖
!pip install torch torchvision einops ema-pytorch accelerate tqdm pillow

# Cell 2: 通用 imports
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from functools import partial
from collections import namedtuple

# Cell 3: GPU 检查
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 学习建议

1. **先读后写**: 每个模块先通读一遍原代码，理解逻辑后再抄写
2. **加注释**: 抄写时用自己的话加中文注释，确保真正理解
3. **画图辅助**: 对 U-Net 结构、扩散过程用图示辅助理解
4. **逐步验证**: 每个模块写完后运行验证代码，确认形状正确
5. **对比实验**: 例如对比不同 beta schedule、不同 attention 类型的效果
6. **不要跳步**: Session 5 是核心，要花最多时间，确保数学公式都理解
