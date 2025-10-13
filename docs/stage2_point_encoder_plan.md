# 点云编码器规划

## 1. 目标
- 实现一个轻量级点云特征提取器，可在 RTX 4060 上实时推理。
- 输出定长特征向量（默认 256 维，可配置），供融合模块使用。
- 支持多帧批处理（batch size ≥ 1），兼容可变点数输入。

## 2. 模型结构
- 简化 PointNet：
  1. 输入 B × N × C（C = 3 或 4）。
  2. 多层 MLP（Linear + BN + ReLU），通道数：64 → 128 → 256。
  3. 全局 Max Pooling 得到 B × 256。
  4. 可选：再经过一层 Linear + ReLU 输出指定维度。
- 利用批归一化提升收敛速度，保持轻量特性。

## 3. 实现细节
- 文件位置：`networks/point_encoder.py`。
- 类定义：`class PointEncoder(nn.Module)`。
- 初始化参数：
  - `in_channels`（默认 3）
  - `feature_dims` 列表（如 [64, 128, 256]）
  - `out_dim`（默认 256）
  - 激活、归一化（可复用 `networks.basic_modules` 中的工具）。
- 前向：接受点云张量 `B × N × in_channels`，返回 `B × out_dim`。
- 提供方法 `freeze_bn()`，便于推理阶段固定 BN。

## 4. 多模态接口
- 输出与视觉/IMU 特征类似（`torch.Tensor` B × out_dim）。
- 后续将在 `fusion_module.py` 中接入此向量。若 `pc_valid=False`，需返回零向量或跳过。

## 5. 测试与验证
- 编写单元测试 `tests/test_point_encoder.py`（暂列 TODO）。
- 在推理脚本（后续任务）中验证维度匹配与 dtype。

## 6. 扩展计划
- 保留 hook，未来可替换为更复杂的点云模型（MinkowskiNet、RandLA-Net）。
- 可选实现 dropout、残差块等提升特征表达。
