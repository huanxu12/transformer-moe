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
\n## 7. 微调执行指南
- 使用 `train_multimodal.py` 时，可通过下方命令模板快速配置：
  ```bash
  python train_multimodal.py \
      --data_path data \
      --pointcloud_path data/pointclouds \
      --train_sequences 00,01,02,03,04,05,06,07,08 \
      --imu_stats data/imu_stats.json \
      --imu_gravity_axis 2 \
      --imu_gravity_value 9.81 \
      --finetune_checkpoint pretrain_models/multimodal_initial.pth \
      --freeze_visual \
      --learning_rate 5e-5 \
      --num_epochs 5 \
      --output_checkpoint pretrain_models/multimodal_finetuned.pth \
      --log_csv logs/finetune.csv \
      --save_every_epoch 1
  ```
- 关键参数说明：
  - `--train_sequences`：指定参与训练的 KITTI 序列，会自动补零。
  - `--imu_stats` / `--imu_gravity_axis` / `--imu_gravity_value`：加载归一化统计并去除重力偏置，训练/评估需保持一致。
  - `--finetune_checkpoint` / `--resume_optimizer`：加载现有模型与优化器，实现继续训练。
  - `--freeze_visual` / `--freeze_point` / `--freeze_imu`：按需冻结子模态，聚焦特定分支。
  - `--output_checkpoint` 与 `--save_every_epoch`：控制最终及阶段性权重输出。
  - `--log_csv`：输出 `epoch, loss` 记录，便于绘制 loss 曲线。
- 默认 batch size 为 1；若需更大 batch，需要改进 `collate_fn` 后再启用。
## 8. IMU 归一化流程

## 9. 训练优化策略

#### 最新结果：Epoch 12 早停 + ReduceLROnPlateau
- 最优验证位于第 9 epoch（trans_rmse=0.1680 m, rot_rmse=0.0024 rad）。
- 评估（09/10）：trans_rmse=0.1673 m, rot_rmse=0.00243 rad；ATE_rmse=61.36 m, RPE_trans_rmse=0.2849 m, RPE_rot_rmse=0.2406°。
- 相比原始 BotVIO：ATE_rmse 降约 18%，RPE_trans_rmse 降约 24%，RPE_rot_rmse 降约 7%。

### 阶段一：监控与早停
- 将 KITTI 09/10 作为固定验证集，每个 epoch（或每隔 N 个 epoch）执行 `evaluations/eval_multimodal.py`，记录平移/旋转误差到 `results/history_epochXX.json`。
- 当验证平移 RMSE 连续 3~4 次未下降 ≥1% 或出现回升时提前停止，并保留指标最优的 checkpoint。
### 阶段二：学习率调度
- 初始学习率保持 5e-5 训练约 10 个 epoch；若验证集指标停滞，降至 1e-5，再视情况调整至 5e-6。
- 可替换为 `ReduceLROnPlateau` 或分阶段重新启动训练脚本以实现多段学习率。
### 阶段三：正则化增强（后续）
- 适当提高 `weight_decay`、在 `Trans_Fusion`/`PoseRegressor` 中引入 Dropout，评估对验证集的影响。
- 尝试冻结部分编码器层以防止过拟合。
### 阶段四：数据增强（后续）
- 扩展图像、点云与 IMU 的随机扰动，模拟真实传感器噪声。
### 阶段五：实验管理（后续）
- 每次策略调整单独记录日志/权重目录，并保留轨迹图用于结果对比。

- 使用 `_tmp_compute_imu.py` 计算均值与方差：例如 `python _tmp_compute_imu.py --data_path data --train_sequences 00,01,02,03,04,05,06,07,08 --imu_gravity_axis 2 --imu_gravity_value 9.81`，脚本会在写入 `data/imu_stats.json` 前对归一化后的均值/方差做断言。
- 训练时为 `train_multimodal.py` 追加 `--imu_stats data/imu_stats.json`，如需去除重力偏置再传入 `--imu_gravity_axis 2 --imu_gravity_value 9.81`。
- 评估脚本 `evaluations/eval_multimodal.py`、`evaluate_pose_multimodal.py` 同样接受以上参数，确保推理阶段使用相同的归一化配置。
- 若需要重新生成统计值，删除旧的 `imu_stats.json` 后重复第一步。`_tmp_compute_imu.py --limit` 可用于抽样验证流程是否正常。
### 阶段三：正则化增强（待执行）
- weight decay：在现有 1e-2 基础上试验 3e-2 / 5e-2，观察验证平移 RMSE 变化，保留最优值。
- Dropout：
  1. 在 `Trans_Fusion` 初始化传入 `drop_prob=0.1`；
  2. 在 `PoseRegressor` 的两层全连接之间插入 `nn.Dropout(0.1)`；
  3. 逐步增大至 0.2，评估验证指标。
- 编码器冻结：
  - 初始阶段冻结视觉编码器前两层；
  - 视验证集表现，尝试同时冻结 PointEncoder 的前两层，以降低过拟合。
- 实验流程：单次仅改一个因素，沿用当前验证配置（09/10），记录 `results/val_history/` 与 `results_finetune_*` 指标。最佳组合固化为新基线。
