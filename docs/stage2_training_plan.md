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
### 阶段三：正则化实验执行
- `weight_decay` 调参：
  - 在 `--weight_decay` 基础值 1e-2 上追加 3e-2、5e-2 轮次，保持其他超参不变。
  - 每轮结束后追踪 `results/val_history/epochXXX.json` 与 `logs/finetune.csv` 的 `trans_rmse` 记录，整理到阶段三表格。
- Dropout 试验：
  - `Trans_Fusion` 通过 `--fusion_drop_prob` 控制；先设 0.1，如验证收益稳定再提升至 0.2。
  - `PoseRegressor` 使用 `--pose_dropout` 在两层全连接之间插入 Dropout，同步递增与融合侧保持一致。
  - 单次实验仅调整一个 Dropout 值，复用其余配置，便于定位收益来源。
- 编码器前段冻结：
  - 视觉编码器用 `--freeze_visual_blocks 2` 冻结前两个卷积块，若验证指标继续改善，再以 `--freeze_point_layers 1` 冻结点云编码器首段进行对比。
  - 如需完整冻结仍可启用 `--freeze_visual`、`--freeze_point`，用于与部分冻结差异分析。
- 实验节奏：
  1. 在阶段二的 09/10 验证方案上复现 baseline（ReduceLROnPlateau+早停保持不变）。
  2. 依次完成 weight_decay 扫描 → Dropout 递增 → 编码器冻结，期间每次仅变动一个因素。
  3. 运行结束后更新 `logs/finetune.csv`、`results/val_history/`，并在阶段计划中记录最佳组合与改进幅度。
- 命令示例（根据需要替换可调参数）：
  ```bash
  python train_multimodal.py \
      --data_path data \
      --pointcloud_path data/pointclouds \
      --train_sequences 00,01,02,03,04,05,06,07,08 \
      --val_sequences 09,10 \
      --finetune_checkpoint pretrain_models/multimodal_initial.pth \
      --output_checkpoint pretrain_models/multimodal_stage3.pth \
      --log_csv logs/stage3_wd003.csv \
      --val_metrics_dir results/val_history \
      --best_checkpoint pretrain_models/multimodal_stage3_best.pth \
      --num_epochs 15 \
      --weight_decay 0.03 \
      --fusion_drop_prob 0.1 \
      --pose_dropout 0.0 \
      --freeze_visual_blocks 2
  ```
  在 Dropout、冻结实验中，只改动对应参数（如 `--fusion_drop_prob 0.2` 或 `--freeze_point_layers 1`），其余保持基准设置，便于快速比对验证指标。

- 验证/里程计评估命令：
  ```bash
  python evaluations/evaluate_pose_multimodal.py \
      --data_path data \
      --pointcloud_path data/pointclouds \
      --eval_sequences 09,10 \
      --checkpoint_path pretrain_models/multimodal_stage3_baseline_best.pth \
      --results_dir results/stage3_baseline_traj \
      --overwrite_results
  ```
  ```bash
  python evaluations/eval_odom.py \
      --data_path data \
      --pred_dir results/stage3_baseline_traj \
      --sequences 09,10 \
      --json_output results/stage3_baseline_odom.json
  ```
- 基线复现（2025-10-15）：
  - 验证最佳：epoch 14，trans_rmse=0.1451 m，rot_rmse=0.00278 rad（results/val_history/epoch014.json）。
  - 里程计：ATE_rmse=36.80 m，RPE_trans_rmse=0.2506 m，RPE_rot_rmse=0.2702°（results/stage3_baseline_odom.json）。

- weight_decay=0.03 对比：
  - 验证最佳：epoch 12，trans_rmse=0.1540 m，rot_rmse=0.0030 rad（logs/stage3_wd003.csv, results/val_history/epoch012.json）。
  - 里程计：ATE_rmse=75.58 m，RPE_trans_rmse=0.2667 m，RPE_rot_rmse=0.2847°（results/stage3_wd003_odom.json）。
  - 结论：相对基线平移/里程计误差增大，暂不列入候选组合。

- weight_decay=0.015 对比：
  - 验证最佳：epoch 12，trans_rmse=0.1673 m，rot_rmse=0.0030 rad（logs/stage3_wd0015.csv, results/val_history/epoch012.json）。
  - 里程计：ATE_rmse=58.49 m，RPE_trans_rmse=0.2877 m，RPE_rot_rmse=0.2974°（results/stage3_wd0015_odom.json）。
  - 结论：较基线平移/里程计误差均上升，效果不佳。

- weight_decay=0.02 对比：
  - 验证最佳：epoch 14，trans_rmse=0.1419 m，rot_rmse=0.0034 rad（logs/stage3_wd002.csv, results/val_history/epoch014.json）。
  - 里程计：ATE_rmse=151.31 m，RPE_trans_rmse=0.2450 m，RPE_rot_rmse=0.3316°（results/stage3_wd002_odom.json）。
  - 结论：轨迹偏差显著放大，判定为失败实验。

- fusion_drop_prob=0.10 试验：
  - 验证：epoch 11 trans_rmse=0.1500 m，rot_rmse=0.0032 rad（pretrain_models/multimodal_stage3_fdrop010_best.pth）
  - 里程计：seq09 ATE=79.14 m、seq10 ATE=16.36 m，总体 ATE_rmse=52.14 m，RPE_trans=0.2608 m，RPE_rot=0.316°（results/stage3_fdrop010_odom.json）
  - 结论：相较 baseline 有轻微退化，可作为 fusion Dropout 上限使用，不建议继续拉高取值。

- fusion_drop_prob=0.15 试验：
  - 验证：epoch 5 trans_rmse=0.1583 m，rot_rmse=0.0039 rad（pretrain_models/multimodal_stage3_fdrop015_best.pth）
  - 里程计：seq09 ATE=85.56 m、seq10 ATE=57.63 m，总体 ATE_rmse=73.55 m，RPE_trans=0.2735 m，RPE_rot=0.379°（results/stage3_fdrop015_odom.json）
  - 结论：高 Dropout 导致 ATE 剧增，判定为失败实验。

- fusion_drop_prob=0.20 试验：
  - 验证：epoch 13 trans_rmse=0.1701 m，rot_rmse=0.0034 rad（pretrain_models/multimodal_stage3_fdrop020_best.pth）
  - 里程计：seq09 ATE=52.60 m、seq10 ATE=57.07 m，总体 ATE_rmse=54.52 m，RPE_trans=0.2963 m，RPE_rot=0.328°（results/stage3_fdrop020_odom.json）
  - 结论：无性能提升且 ATE 仍偏高，判定为失败实验。

