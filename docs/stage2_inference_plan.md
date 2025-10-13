# 推理脚本补建规划（阶段2-T2.4）

## 目标
- 结合现有数据集与新增点云编码器/融合模块，重建多模态推理脚本骨架。
- 完成数据加载、特征提取（含点云）、融合流程，验证前向传播无 NaN。
- 由于原仓库缺失视觉/IMU 编码器与评估脚本，本阶段以可运行的占位实现替代，后续待补全真实网络与权重。

## 设计
1. 新增脚本 `evaluations/eval_multimodal.py`：
   - 解析命令行参数（复用 `BotVIOOptions`）。
   - 构建 `KITTIOdomPointDataset` 数据加载器。
   - 初始化编码器：
     - 视觉编码器：当前缺失，使用轻量占位模块 `DummyVisualEncoder`（卷积 + 池化），后续替换。
     - IMU 编码器：使用简单 MLP 占位模块 `DummyIMUEncoder`，后续替换。
     - 点云编码器：使用 `networks.PointEncoder`。
     - 融合模块：使用改造后的 `Trans_Fusion`，支持点云。
   - 迭代数据并执行前向传播，输出特征统计/占位位姿（例如恒等矩阵），确保无异常。
   - 将样例输出保存至 `results/debug_multimodal.pkl`，便于后续调试。

2. 占位模块说明：
   - 位于 `networks/dummy_modules.py`，提供简单可运行的编码器，后续替换为真实实现。
   - 明确 TODO，避免与正式模块混淆。

3. 兼容性与扩展：
   - 当真实视觉/IMU 编码器接入后，替换占位模块即可。
   - 推理脚本保留接口 `--pointcloud_path`，支持无点云模式（自动跳过）。

4. 验证方法：
   - 运行 `python evaluations/eval_multimodal.py --pointcloud_path=data/pointclouds`，检查输出维度与日志。
   - 确认 `results/` 目录生成调试文件，且运行无 NaN。

5. 后续工作：
   - 待真实网络实现后，替换占位模块并接入已有评估流程（ATE/RPE）。
   - 根据计划 T2.5 做联合微调，T2.6 输出阶段报告。
