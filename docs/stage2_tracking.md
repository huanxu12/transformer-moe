# 阶段2 多模态融合执行跟踪

## 目标
- 在 KITTI 里程计数据上验证视觉 + 点云 + IMU 三模态融合的稳定性。
- 形成可复现的评估工具链（推理、可视化、误差统计）。

## 任务状态
- [完成] T2.1 数据集结构扩展：`datasets/kitti_point_dataset.py` 支持点云读取与参数化裁剪，对应 CLI 选项已在 `evaluations/options.py` 对齐。
- [完成] T2.2 点云编码器实现：`networks/point_encoder.py` 输出 256 维特征，并提供 BN 冻结接口。
- [完成] T2.3 融合模块更新：`networks/fusion_module.py` 接入点云特征及 `pc_valid`，完成门控加权。
- [推进中] T2.5 预训练/微调：`train_multimodal.py` 已支持加载旧权重、冻结分支、阶段性保存与 CSV 记录；下一步需在真实训练上验证并补充批量化 collate。
- [完成] T2.6 阶段评估：`evaluate_pose_multimodal.py`、`plot_trajectories.py`、`eval_odom.py` 均已落地并通过 09/10 序列验证。

## 评估记录（2025-10-11）
- 推理：`python evaluations/evaluate_pose_multimodal.py --data_path data --pointcloud_path data/pointclouds --num_workers 0 --eval_sequences 09,10 --checkpoint_path pretrain_models/multimodal_initial.pth --results_dir results --overwrite_results`
- 误差指标（`results/metrics.json`）：
  - 序列 09：ATE RMSE 131.35 m，RPE trans RMSE 0.34 m，RPE rot RMSE 0.44°。
  - 序列 10：ATE RMSE 74.37 m，RPE trans RMSE 0.39 m，RPE rot RMSE 0.46°。
- 轨迹图：`results/plots/09.png`、`results/plots/10.png`（Sim(3) 对齐后仍存在明显漂移）。
- 日志提示：首次加载模型仍会显示 `torch.load(..., weights_only=False)` 的未来行为变更，可在权重可信的前提下保持现状。

## 高 ATE 诊断
- 虽然逐帧 RPE（平移 ~0.35 m，旋转 ~0.45°）较小，但累计姿态存在显著偏差：Sim(3) 对齐后序列 09 末帧位置仍较 GT 偏移约 238 m。
- 统计显示预测平均航向角与 GT 的最终差异约 112°，说明存在持续的小角度偏差，长距离积分后显著放大。
- 预测步长均值（0.85 m）低于 GT（1.07 m），说明尺度仍偏小，进一步削弱对齐效果。
- 综合判断：高 ATE 主要由“微小旋转偏差 + 步长低估”叠加导致的轨迹漂移，可通过改进姿态正则、引入闭环/重定位约束或追加 scale 校准来缓解。

## 后续待办
- 按 T2.5 规划继续补充微调/再训练实验，重点观察 yaw & scale 偏差的收敛情况。
- 结合 yaw drift 数据，评估在损失中引入姿态稳定项或增强 IMU 对齐的必要性。
- 扩展 `plot_trajectories.py`：计划新增 3D 视角或误差热力图，以辅助定位漂移阶段。
