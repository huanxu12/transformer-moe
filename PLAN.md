# BotVIO 多模态语义 SLAM 开发计划

## 1. 项目目标
- 在保留现有 BotVIO 框架的基础上，拓展为图像 + 点云 + IMU 的多模态语义 SLAM 系统。
- 支持在 KITTI（离线）、DARPA、自采 .bag 数据上运行，逐步迈向在线/实时能力。
- 评估指标需优于当前公开 VO/VIO 结果，并引入语义地图相关指标。
- 工程实践中严格遵循计划文档，确保需求、数据、开发、测试全流程可追溯。

## 2. 阶段划分与里程碑
### 阶段 1：调研与环境铺设（第 1-2 周）
目标：明确语义 SLAM 功能边界、复用模块、风险点，完成工具与协作方案。
- T1.1：整理语义 SLAM 功能需求、语义地图形式、实时/离线要求。
- T1.2：梳理现有代码模块，标记可复用和需改动部分。
- T1.3：调研点云编码器、语义分割模型、语义地图构建方案，形成调研报告。
- T1.4：规划数据来源与解析流程，包含 KITTI、DARPA、自采 .bag 的读取、同步方式。
- T1.5：搭建协作与实验记录机制，规划日志、版本库、文档结构。
- T1.6：确认环境依赖及脚本（安装说明、数据准备脚本、单位测试环境）。

### 阶段 2：三模态基础融合（第 3-6 周）
目标：接入点云数据，实现图像+点云+IMU 特征级融合与初步评估。
- T2.1：扩展数据集结构，引入点云读取、预处理、降采样支持。
- T2.2：实现点云编码器（PointNet++/轻量 BEV 等），输出定长特征。
- T2.3：改造融合模块，支持三模态特征交互；更新默认参数与配置。
- T2.4：更新推理脚本（VO/VIO），完成三模态前向；确保输出无 NaN。
- T2.5：根据需要对点云分支或整体模型在 KITTI 上做微调，输出初版权重。
- T2.6：完成阶段评估，形成初步结果与报告。

### 阶段 3：语义增强与地图构建（第 7-10 周）
目标：引入语义信息，构建语义地图并完成语义评估。
- T3.1：集成图像或点云语义分割模型，生成语义标签。
- T3.2：设计并实现语义地图结构（语义点云/语义栅格/稀疏网格）。
- T3.3：利用语义信息优化位姿估计、回环检测或关键帧管理。
- T3.4：扩展数据解析管线以兼容 DARPA、自采 .bag，完成语义标签对齐。
- T3.5：制定并执行语义指标评估（IoU、语义保持率、语义漂移等）。
- T3.6：产出语义增强模块的详细文档与实验报告。

### 阶段 4：实时化与系统整合（第 11-16 周）
目标：提升性能、整合系统，完成跨数据集验证与交付。
- T4.1：性能 Profiling，优化数据加载与推理（混合精度、点云稀疏化、异步 IO 等）。
- T4.2：设计在线架构（ROS/ROS2 或其他消息机制），实现实时传感器接入。
- T4.3：在 KITTI、DARPA、自采数据上完成全流程评估，输出对比表。
- T4.4：整理交付物（代码、脚本、说明文档、使用指南）。
- T4.5：总结风险与优化建议，为后续迭代提供依据。

## 3. 数据与预训练策略
- 图像、IMU：继续沿用现有权重；如网络结构变化，需在 KITTI 上微调。
- 点云：优先使用已有模型权重（PointNet++、MinkowskiNet 等）；如无可用权重，先在 KITTI 上训练初版，再联合微调。
- 语义模型：图像语义选择 CITYSCAPES/KITTI 预训练模型；点云语义参考 SemanticKITTI 模型。
- 联合训练：三模态融合后建议端到端微调，提升特征对齐。

### 3.1 现阶段推荐的多模态权重
- **pretrain_models/multimodal_initial.pth**（Stage 2 端到端预训练，pose_dropout=0.02）
  - 2025-10-20 评估：Seq09 ATE=44.34 m，Seq10 ATE=39.60 m，Overall ATE=42.30 m；RPE_trans=0.274 m，RPE_rot=0.250 deg（`results/diagnostics_initial_eval/metrics_odom.json`）。
  - 作为当前默认起点，后续调度/微调实验优先采用此权重。
- **pretrain_models/multimodal_baseline_best.pth**（Stage 3 baseline，pose_dropout=0）
  - 2025-10-20 评估：Seq09 ATE=72.84 m，Seq10 ATE=40.29 m，Overall ATE=58.84 m；RPE_trans=0.263 m，RPE_rot=0.208 deg（`results/diagnostics_baseline_best_eval/metrics_odom.json`）。
  - 用于与历史 Stage 3 结果对齐或做 pose dropout=0 的对照。
- **不再使用** `pretrain_models/multimodal_initial_epoch3.pth`：2025-10-20 评估 Overall ATE≈116 m，存在严重漂移，仅保留为异常样本。

## 4. 评估体系
- 几何指标：ATE、RPE、平移/旋转误差，对比现有 VO/VIO 结果。
- 语义指标：语义 IoU、语义保持率、回环语义一致性。
- 实时性：帧率、平均延迟、GPU/CPU 占用。
- 消融实验：逐一关闭模态分支或语义模块，分析贡献。

## 5. 协作与记录
- 版本管理：使用 Git 分支 + PR，关键改动需评审。
- 实验记录：统一表格/文档记录数据源、参数、指标、生成时间等。
- 文档体系：
  - `PLAN.md`：总规划文件（仅由负责人更新）。
  - `docs/stage*_tracking.md`：阶段任务跟踪。
  - `docs/data_guides/`、`docs/models/` 等：数据、模型、语义模块说明。
- 里程碑评审：每阶段结束需提交报告，审阅后方可进入下一阶段。

## 6. 风险与应对
- 数据差异大：提前设计泛化良好的数据接口，保证不同源数据可切换。
- 点云计算开销高：采用采样、稀疏卷积优化，必要时规划多线程。
- 语义模型负担重：选用轻量模型或分类精度与实时性兼顾的网络。
- 训练资源不足：优先微调，必要时寻求更高算力或分布式方案。
- 时间偏差：如某任务延迟，应在阶段跟踪文档中注明原因与调整计划。

## 7. 执行约束
- 每项任务完成后必须对照本计划检查是否偏离；如有偏差需要在跟踪文档中记录原因和应对策略。
- 开发、测试、文档更新需同步进行，确保可重现性。
- 任何超出计划的工作需评估后再执行，避免资源分散。

## 8. 下一步计划 (2025-10-16)
1. 针对 `stage3_pose_drop003` 进行稳定性复现：增补不同随机种子与更多 KITTI 序列（含 00-08 子集），验证指标波动并记录运行时统计。
- stage3_pose_dropout_schedule 自适应：pose_dropout 0.15->0.03->0 使 overall ATE≈46.5 m，Seq09 帧 380-520 仍有约 150-190 m 漂移，需结合 `results/analysis/seq09_drop_segments.csv` 与抽样数据排查。
2. 对 `stage3_pose_drop004` 与 `stage3_pose_drop006` 的 09/10 轨迹做误差热力与段落化对比，定位 RMSE 峰值区间，并核对对应的原始传感器输入。
3. 在训练脚本中加入 dropout 自适应策略候选（基于姿态不确定度或梯度阈值），并设计小规模消融验证是否能兼顾验证集与 odom 表现。
4. 清理评估流程中的告警：替换 `ReduceLROnPlateau` 的 `verbose` 旧参数；在 `torch.load` 调用中启用 `weights_only=True` 并测试兼容性，降低未来版本风险。
5. 建立阶段性实验看板：将本周 odom 结果整理入 `results/metrics_history.csv`，并扩充脚本以自动生成 drop 系列曲线图，支撑后续汇报。

## Appendix: IMU Normalization Plan
1. Compute IMU mean/std on training sequences and save them to data/imu_stats.json (keep the command template in README/tools).
2. Add --imu_stats, --imu_gravity_axis, --imu_gravity_value options in BotVIOOptions and pass them during training/evaluation runs.
3. Apply (imu - mean) / std inside prepare_batch; subtract the gravity constant from the configured axis when needed.
4. To roll back, remove the CLI options, delete the normalization block, and drop data/imu_stats.json.

## Appendix: Training Optimization Roadmap
### Latest Run Summary (Epoch 12 early stop, ReduceLROnPlateau)
- Validation best at epoch 9: trans_rmse=0.1680 m, rot_rmse=0.0024 rad (logs/finetune.csv, results/val_history/epoch009.json).
- Final evaluation (pretrain_models/multimodal_finetuned_best.pth) on 09/10: trans_rmse=0.1673 m, rot_rmse=0.00243 rad (results/finetune_eval_metrics.json).
- Odom metrics after evaluate_pose_multimodal + eval_odom: ATE_rmse=61.36 m, RPE_trans_rmse=0.2849 m, RPE_rot_rmse=0.2406 deg (results_finetune_best/metrics_odom.json).
- Compared to original BotVIO (results_finetune_fusion/metrics_finetune.json): ATE_rmse ↓13.9 m (~18%), RPE_trans_rmse ↓0.0897 m (~24%), RPE_rot_rmse ↓0.018 deg (~7%).

- Stage 3 baseline (2025-10-19, 15 epochs, pose_dropout=0): validation best at epoch 6 with trans_rmse=0.1539 m, rot_rmse=0.0021 rad (logs/stage3_baseline.csv); evaluation on KITTI 09/10 yields Seq09 ATE=72.21 m / RPE_trans=0.234 m, Seq10 ATE=39.81 m / RPE_trans=0.303 m, overall ATE=58.27 m, RPE_trans=0.264 m (results/stage3_baseline/metrics_odom.json).

### Stage 3 Pose Dropout Sweep (2025-10-16 更新)
| 实验 | Seq09 ATE RMSE (m) | Seq10 ATE RMSE (m) | Overall ATE RMSE (m) | Overall RPE_trans (m) | Overall RPE_rot (deg) | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| stage3_pose_drop001 | 209.48 | 108.52 | 166.05 | 0.266 | 0.361 | 基线 drop=0.01，整体漂移明显。 |
| stage3_pose_drop002 | 64.25 | 62.76 | 63.61 | 0.393 | 0.392 | RPE 增大，整体仍优于其他设置。 |
| stage3_pose_drop003 | 50.72 | 40.54 | 46.34 | 0.276 | 0.303 | 当前最优，误差分布稳定。 |
| stage3_pose_drop004 | 120.71 | 59.63 | 94.44 | 0.273 | 0.424 | 中位误差低但存在灾难性偏移。 |
| stage3_pose_drop005 | 122.98 | 84.80 | 106.56 | 0.272 | 0.369 | 适中丢弃率但收益有限。 |
| stage3_pose_drop006 | 241.15 | 150.01 | 201.94 | 0.404 | 0.387 | 高丢弃率导致轨迹发散。 |
| stage3_pose_drop007 | 88.82 | 91.04 | 89.78 | 0.278 | 0.449 | 双序列误差接近，局部抖动偏大。 |
| stage3_pose_drop008 | 213.87 | 117.03 | 172.21 | 0.272 | 0.491 | 丢弃更高亦无收益。 |
| stage3_pose_drop025 | 218.04 | 85.08 | 160.85 | 0.359 | 0.370 | 训练验证尚可但实测 ATE 高。 |
| stage3_pose_drop030 | 239.32 | 116.47 | 186.47 | 0.272 | 0.400 | 验证优秀但实际轨迹严重漂移。 |
| stage3_pose_drop035 | 256.00 | 110.61 | 193.46 | 0.423 | 0.381 | 进一步增大 dropout 完全失效。 |
| stage3_pose_dropout_schedule | 52.66 | 38.38 | 46.52 | 0.257 | 0.248 | 自适应调度 0.15->0.03->0，Seq09 中段偏移待修复 |
- `stage3_pose_drop003` 仍是当前最佳设定，建议作为阶段性基线保留。
- drop≥0.25 的实验在验证集 RMSE 较低，但 odom 误差失控，需检查关键信息缺失与评估脱节问题。
- 建议针对 `stage3_pose_drop004`、`stage3_pose_drop006` 做局部轨迹对齐分析，定位异常片段。

1. Phase 1 (Early stopping): enable validation on KITTI 09/10 each epoch, persist metrics under `results/history_epochXX.json`, and stop when translation RMSE fails to improve beyond 1% for 3 consecutive evaluations.
2. Phase 2 (Learning-rate scheduling): switch from batch-wise cosine decay to validation-driven `ReduceLROnPlateau`; decay LR by factor 0.3 after 2 stagnant validations and continue training with checkpoints for each epoch.
3. Phase 3 (Regularization, in progress): leverage `--weight_decay`, `--fusion_drop_prob`, `--pose_dropout`, and encoder freezing flags (`--freeze_visual_blocks`, `--freeze_point_layers`) to run single-factor regularization sweeps under KITTI 09/10 validation logging.
   - Dropout sweep (2025-10-15):
     - fusion_drop_prob=0.10 → 验证 trans_rmse≈0.150 m、ATE≈52 m，略逊于基线。
     - fusion_drop_prob=0.15/0.20 → ATE >70 m，判定失败。
     - 下一步聚焦 PoseRegressor Dropout、编码器局部冻结等其他正则化手段。
4. Phase 4 (Data augmentation, TODO): extend visual/point/IMU augmentations to simulate realistic noise and improve robustness.
5. Phase 5 (Experiment management, TODO): formalize run naming, metric aggregation, and trajectory plotting comparisons for reproducible reporting.

### 2025-10-16 实施\n- 新增 	ools/ensure_kitti_velodyne.py，用于检查/补齐 Seq09/10 的 velodyne 数据（支持 --dry-run、--src-root、--zip）。\n- 	rain_multimodal.py 支持 --pose_dropout_schedule 自适应调整；日志记录每 epoch 的实际数值。\n- MonoDataset/KITTIOdomPointDataset 扩展遮挡与 LiDAR sector dropout 增强，通过 --aug_image_occlusion_*、--aug_lidar_sector_dropout_* 控制。\n

## 9. 应用场景环境假设
- 目标部署：地下灾害后的矿井，预计存在极端弱光、多尘、多烟环境。
- 视觉挑战：纹理稀疏、结构重复、强自遮挡与动态干扰，需要在模型训练与评估中模拟。
- 感知策略：
  - 训练阶段保留或增强点云/IMU 约束，避免在弱纹理场景中过度依赖视觉。
  - 数据增广时重点覆盖遮挡、雾霾、低照度与点云稀疏等情况。
  - 评估环节增加对高遮挡/低纹理样本的专门回放，以验证鲁棒性。
## 10. 地下矿井极端环境专项计划（2025 Q4 - 2026 Q2）
### 10.1 数据与增广准备（2025-10 ~ 2025-12）
- 汇集现有地下矿井/灾害演练数据，补充采集夜视、低照度、烟尘路段，建立多模态时间同步基准。
- 构建视觉增广 pipeline：低照度、雾霾/烟雾、局部遮挡、纹理稀疏仿真，并记录增广参数以便复现。
- 点云端模拟粉尘稀疏、扇区遮挡及多路径噪声，配合 IMU 噪声/零偏漂移脚本，形成极端工况组合表。
- 制定数据质量审计清单（时间同步、外参稳定性、传感器健康度）作为批处理前置条件。

### 10.2 感知模型稳健化（2025-11 ~ 2026-01）
- 在 MultimodalModel 中强化点云/IMU 约束：引入单模态姿态辅助损失与点云 gating 正则，防止视觉独断。
- 训练阶段加入传感器置信度指标（亮度、有效点数、IMU 方差），驱动 Trans_Fusion 门控动态调节权重。
- 针对连续视觉退化样本执行 curriculum 学习（先短窗、再长窗），观察跨模态补偿能力。
- 记录 ablation：仅视觉、视觉+IMU、全模态，形成弱纹理 benchmark 对比。

### 10.3 评估与回放体系（2025-12 ~ 2026-02）
- 搭建“高遮挡/低纹理”评估子集，按场景类型（粉尘、烟雾、湿滑镜面）分层统计 ATE/RPE。
- 引入多模态健康监控：离线回放时落地传感器 dropout、噪声突增检测，并生成事件时间线。
- 构建极端工况 KPI（鲁棒性评分、失败恢复时间、单模态 fallback 质量），纳入阶段性报告模板。
- 自动化回放脚本支持批量注入遮挡/噪声扰动，比较有无稳健化策略的误差差异。

### 10.4 MoE 混合专家引入路线（2025-12 ~ 2026-03）
- 设计专家划分：按传感器可用性（视觉受损、点云稀疏、全模态正常）设定三类专家，复用现有编码器权重。
- 开发轻量 router（输入传感器置信度与历史误差），采用软路由 + 负载均衡损失，先在离线数据上预训练。
- 逐步上线：先冻结视觉专家、训练“点云优先生效”路线 → 解冻联合训练 → 对比统一模型。
- 在评估子集上完成 MoE 与单体模型的 A/B 实验，并记录额外算力、延迟与稳定性指标。
