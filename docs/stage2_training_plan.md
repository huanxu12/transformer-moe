# T2.5 训练/微调规划

## 1. 目标
- 基于现有三模态骨架，完成一次初步的微调或训练，使视觉、IMU、点云特征对齐。
- 产出可在 KITTI 上运行的初版权重，为后续评估（T2.6）做准备。

## 2. 前置条件
- 真实视觉/IMU 编码器权重尚缺，需要确定替代方案：
  1. 使用 Dummy 编码器进行预训练，待真实编码器接入后再微调。
  2. 或寻找开源视觉/IMU 模型，迁移到当前框架。
- 数据集暂以 KITTI 为起点，batch size 受限于 GPU (4060)。

## 3. 训练策略建议
- 视觉分支：
  - 若无预训练权重，可使用轻量卷积或借鉴现有 Vision Transformer 编码器。
  - 可采用自监督或对比学习，保持图像特征稳定。
- 点云分支：
  - 初期使用随机初始化 PointEncoder，配合其他模态进行联合训练。
  - 可加入点云掩码/随机旋转等增强。
- IMU 分支：
  - 可使用 MLP 或 GRU 结构；对 IMU 数据进行归一化。
- 融合模块：
  - 启用点云门控与 Dropout，提升融合鲁棒性。

## 4. 训练流程
1. 构建训练脚本 `train_multimodal.py`（TODO）：
   - 数据加载：KITTIOdomPointDataset，含图像、点云、IMU。
   - 模型：视觉/IMU/点云编码器 + 融合模块 + 简易位姿解码器。
   - 损失函数：
     - 监督：与 GT 位姿对齐（ATE/RPE 方向）或使用 L1/L2。
     - 自监督：基于光流/重投影误差（若图像质量允许）。
2. 训练参数：
   - Epoch：1~5（探索性），Batch size：1~2。
   - Optimizer：AdamW，学习率 1e-4。
   - 保存权重到 `pretrain_models/`，命名 `multimodal_initial.pth`。
3. 日志记录：
   - TensorBoard 或 CSV 记录 loss、精度。
   - 每轮保存模型与配置。

## 5. 风险与补救
- 无真实编码器：先用占位模型，评估趋势。待真实模型接入后重复微调。
- 数据缺失（点云/IMU 不匹配）：需要在训练脚本中加入数据完整性检查。
- GPU 显存不足：控制点云点数、分批处理。

## 6. 输出要求
- 记录训练命令、参数、日志路径。
- 保存模型权重并更新 README/文档说明。
- 在 `docs/stage2_tracking.md` 更新 T2.5 状态及备注。
\n## 7. ΢调执行指南\n- 使用 `train_multimodal.py` 时，可通过新参数快速配置：\n  ```bash\n  python train_multimodal.py \\\n      --data_path data \\\n      --pointcloud_path data/pointclouds \\\n      --train_sequences 00,01,02,03,04,05,06,07,08 \\\n      --finetune_checkpoint pretrain_models/multimodal_initial.pth \\\n      --freeze_visual \\\n      --learning_rate 5e-5 \\\n      --num_epochs 5 \\\n      --output_checkpoint pretrain_models/multimodal_finetuned.pth \\\n      --log_csv logs/finetune.csv \\\n      --save_every_epoch 1\n  ```\n- 关键参数说明：\n  - `--train_sequences`：指定参与训练的 KITTI 序列，自动补零。\n  - `--finetune_checkpoint` / `--resume_optimizer`：加载现有模型和优化器状态，实现继续训练。\n  - `--freeze_visual` / `--freeze_point` / `--freeze_imu`：按需冻结子模态，聚焦特定分支。\n  - `--output_checkpoint` 与 `--save_every_epoch`：控制最终及阶段性权重输出。\n  - `--log_csv`：输出 `epoch, loss` 记录，便于后续绘制 loss 曲线。\n- 默认为 batch size 1；若需要更高 batch size，需改进 `collate_fn` 后再开启。\n
