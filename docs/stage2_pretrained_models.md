# 预训练模型与评估脚本说明

## 1. 新增模型实现
- `networks/visual_encoder.py`：轻量 CNN 编码器，输出 512 维视觉特征。
- `networks/imu_encoder.py`：GRU 基于的 IMU 编码器，输出 256 维特征。
- `networks/point_encoder.py`：点云特征提取器（既有）。
- `networks/pose_regressor.py`：将融合特征映射到 6-DoF 位姿（轴角 + 平移）。
- `networks/fusion_module.py`：支持视觉 + 点云 + IMU 融合，并带门控。

## 2. 训练脚本 `train_multimodal.py`
- 解析 `BotVIOOptions` 参数，加载 `KITTIOdomPointDataset`。
- 构建 `MultimodalModel`（视觉/点云/IMU 编码器 + 融合 + 位姿回归）。
- 使用 KITTI GT 位姿构建 L1 损失，训练后保存 `pretrain_models/multimodal_initial.pth` 与日志。
- 使用方法（示例）：
  ```bash
  python train_multimodal.py \
      --data_path data \
      --pointcloud_path data/pointclouds \
      --num_epochs 1 --learning_rate 1e-4 --num_workers 0
  ```
- 输出：
  - `pretrain_models/multimodal_initial.pth`
  - `pretrain_models/multimodal_training_log.pkl`

## 3. 推理脚本 `evaluations/eval_multimodal.py`
- 载入训练后的权重（默认 `--checkpoint_path pretrain_models/multimodal_initial.pth`）
- 对 `data/` 中的序列进行前向推理，生成 `results/XX.txt`（TODO：待补真实位姿输出）。
- 当前版本验证三模态前向通路是否正常运行，未写入轨迹文件，待下阶段补全。

## 4. 评估脚本（待补）
- `docs/stage2_evaluation_plan.md` 规划了评估流程，后续需要编写 `evaluate_pose_multimodal.py`：
  - 加载模型与权重。
  - 遍历数据集，生成 KITTI 格式轨迹文件 `results/09.txt`、`results/10.txt`。
  - 调用 `eval_odom.py` 计算 ATE/RPE。
- 语义模块接入后扩展语义指标评估。

## 5. 权重管理
- 所有新的权重默认保存到 `pretrain_models/`。
- 建议建立命名规范：`multimodal_<dataset>_<date>.pth`。
- 训练/评估日志应附带参数表、数据来源说明。

## 6. 待办提醒
- 需要安装 PyTorch 等依赖后才能运行训练/推理脚本。
- 等真实大模型或外部预训练权重确定后，替换当前轻量实现。
- 补充 `evaluate_pose_multimodal.py` 与可视化脚本以完成 T2.6。
