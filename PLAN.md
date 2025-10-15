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

1. Phase 1 (Early stopping): enable validation on KITTI 09/10 each epoch, persist metrics under `results/history_epochXX.json`, and stop when translation RMSE fails to improve beyond 1% for 3 consecutive evaluations.
2. Phase 2 (Learning-rate scheduling): switch from batch-wise cosine decay to validation-driven `ReduceLROnPlateau`; decay LR by factor 0.3 after 2 stagnant validations and continue training with checkpoints for each epoch.
3. Phase 3 (Regularization, TODO): experiment with higher `weight_decay`, Dropout in `Trans_Fusion`/`PoseRegressor`, and selective encoder freezing to mitigate overfitting.
4. Phase 4 (Data augmentation, TODO): extend visual/point/IMU augmentations to simulate realistic noise and improve robustness.
5. Phase 5 (Experiment management, TODO): formalize run naming, metric aggregation, and trajectory plotting comparisons for reproducible reporting.

