# 融合模块改造规划

## 1. 目标
- 让 `Trans_Fusion` 支持视觉、IMU、点云三路特征输入。
- 在保证原有 VO/VIO 流程兼容的前提下，新增点云特征路径，可按需启用/禁用。
- 允许对点云特征进行投影/门控，实现柔性融合。

## 2. 设计要点
- 新增构造参数：
  - `point_dim`：点云特征维度（默认与视觉维度一致）。
  - `point_dropout`（可选）：对点云特征的 dropout。
- 内部模块：
  - `self.point_proj`：将点云特征映射到 `self.dim`。
  - `self.point_gate`：两层线性+Sigmoid 门控，调节点云对视觉特征的贡献。
- 前向接口修改：
  ```python
  def forward(self, visual, imu, point=None, point_valid=None):
  ```
  - `point` 为 `B × point_dim` 张量；若为 `None` 或 `point_valid=False`，回退到原有视觉+IMU 逻辑。
  - 若点云存在：
    1. `point_feat = self.point_proj(point)`
    2. `gate = self.point_gate(point_feat)`
    3. `visual = visual + gate * point_feat`
- IMU 分支保持不变，仍使用原有 Cross-Covariance Attention。

## 3. 数值稳定性
- 点云特征与视觉特征在 dtype/device 上保持一致（与输入相同）。
- 若点云特征来自空输入，返回全零向量，并标记 `point_valid=False`，避免对视觉造成影响。

## 4. 兼容性
- 原函数调用仍可传入两个参数；通过默认值 `point=None` 保证旧脚本不报错。
- `point_dim=None` 时自动跳过点云路径。

## 5. 后续任务
- 更新 `evaluate_pose_vio.py` 等推理脚本以正确传入点云特征。
- 在阶段 2 末进行联合评估，验证新增特征的效果。
