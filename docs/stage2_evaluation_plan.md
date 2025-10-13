# 阶段2 评估执行计划

## 1. 评估目标
- 使用当前三模态模型在 KITTI Odom 序列 09/10 上生成并对齐完整轨迹。
- 输出 ATE、RPE 以及平移/旋转误差，并配套轨迹可视化图。
- 确认多模态融合链路具备可重复的离线评估流程。

## 2. 执行步骤
1. 训练/微调：通过 `train_multimodal.py` 获取或更新 `pretrain_models/multimodal_initial.pth`。
2. 轨迹推理（已完成）：运行 `evaluations/evaluate_pose_multimodal.py` 生成 `results/<seq>.txt`。
   ```bash
   python evaluations/evaluate_pose_multimodal.py \
       --data_path data \
       --pointcloud_path data/pointclouds \
       --num_workers 0 \
       --eval_sequences 09,10 \
       --checkpoint_path pretrain_models/multimodal_initial.pth \
       --results_dir results --overwrite_results
   ```
3. 轨迹可视化（已完成）：使用 `evaluations/plot_trajectories.py` 绘制预测与 GT 对比图。
   ```bash
   python evaluations/plot_trajectories.py \
       --data_path data \
       --pred_dir results \
       --sequences 09,10 \
       --output_dir results/plots
   ```
4. 误差评估（已完成）：调用 `evaluations/eval_odom.py` 计算 ATE/RPE，可输出 JSON。
   ```bash
   python evaluations/eval_odom.py \
       --data_path data \
       --pred_dir results \
       --sequences 09,10 \
       --json_output results/metrics.json
   ```
5. 结果整理：
   - 在 `results/` 保留轨迹文件、评估日志、可视化图。
   - 在 `docs/experiments/` 记录对应实验条目（待新增）。
6. 阶段报告：汇总误差曲线、对比基线 VO/VIO，并输出阶段性总结。

## 3. 前置条件
- `data/sequences`, `data/pointclouds`, `data/imus`, `data/poses` 均已完整就绪。
- 评估脚本共享同一 `BotVIOOptions` 配置，参数定义于 `evaluations/options.py`。

## 4. 待补事项
- 若后续引入语义或其他模态，需要在 `eval_odom.py` 扩展指标或输出。
- 轨迹可视化脚本可进一步增加 3D 视角、误差热力等增强功能。

## 5. 输出内容
- ATE/RPE 指标表及（可选）JSON 汇总。
- 轨迹对齐图、误差随时间曲线。
- 关键结论与潜在改进点。

## 6. 下一步
- 完善可视化工具与报告模板。
- 根据需要扩展至更多序列或自采数据，并纳入阶段总结/README。
