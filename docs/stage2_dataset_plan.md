# T2.1 数据集结构扩展设计

## 1. 文件结构调整
- 新增 `data/pointclouds/<seq>/<frame>.bin` 存放点云（KITTI Velodyne 原始格式）。
- `meta/` 目录增加 `pointcloud_timestamps_<seq>.csv`，记录点云时间戳与帧号的映射。
- 为未来 bag 转换生成的点云，约定统一保存为 KITTI 风格的 `float32` BIN（每点 4 个数：x, y, z, intensity）。

## 2. 数据加载策略
- 新建 `datasets/kitti_point_dataset.py`，继承现有 `KITTIOdomDataset`，在 `__getitem__` 中：
  1. 读取图像、IMU 逻辑保持不变。
  2. 新增 `load_pointcloud(seq, frame_id)`，读取 `pointclouds` 目录的对应 BIN 文件。
  3. 执行预处理（见第 3 节），输出张量 `pointcloud`，形状 `N×4` 或 `N×3`。
  4. 在返回的 `inputs` 字典中加入键 `("pointcloud", 0, 0)`。
- 保持原 API（`__getitem__(index)` 返回 `inputs`, `targets`），确保向下兼容。

## 3. 预处理与降采样
- 加载后执行以下步骤：
  - 去除 NaN/Inf 点。
  - 根据需要过滤距离过近/过远的点（例如 0.5m < r < 60m）。
  - 随机/体素采样控制点数（例如保留 8192 点）。
  - 归一化或添加额外特征（如强度、时间戳）。
- 拟新增工具函数到 `util/pointcloud_ops.py`（待创建），封装采样与滤波逻辑。

## 4. 配置与参数
- 在 `options.py` 中新增命令行参数：
  - `--pointcloud_path`：点云根目录（默认 `data/pointclouds`）。
  - `--pc_max_points`：每帧最大点数。
  - `--pc_min_range` / `--pc_max_range`：距离过滤范围。
- 更新 `BotVIOOptions` 默认值，并兼容旧脚本（未传入点云路径时保持原行为）。

## 5. 与下游模块接口
- `inputs` 中的 `("pointcloud", 0, 0)` 将被点云编码器（T2.2）使用。
- 若无点云文件，则根据配置选择：
  - 抛出异常（用于严格模式）。
  - 返回空张量并标记 `inputs["pc_valid"] = False`，以便下游跳过点云分支。

## 6. 实施步骤
1. 创建 `util/pointcloud_ops.py`，编写基础采样函数（占位）。
2. 编写 `datasets/kitti_point_dataset.py`，实现新的数据集类。
3. 在 `datasets/__init__.py` 中注册该数据集。
4. 更新 `options.py` 参数与默认值。
5. 编写单元测试草案 `tests/test_dataset_pointcloud.py`（可先放 TODO）。

## 7. 后续工作
- 等待点云编码器（T2.2）实现，以验证数据输入格式。
- 在 README / 文档中更新点云数据准备说明。
- 实施完成后，整理示例脚本（例如 `tools/prepare_pointclouds.sh`）。
