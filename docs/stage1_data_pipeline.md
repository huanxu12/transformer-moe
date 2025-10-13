# 数据解析流程设计

## 1. 数据源概览
- **KITTI Odometry**：图像（PNG）、激光点云（BIN）、IMU（文本/原KITTI格式）、位姿（TXT）。
- **DARPA / 自采 .bag**：ROS bag，包含图像、点云、IMU、可能的里程计/GPS。
- **其他扩展**：未来可加入深度相机、语义标签等。

## 2. 统一目录结构建议
```
data/
├── sequences/              # 图像序列（与 KITTI 保持一致）
│   ├── 09/image_2/XXXX.png
│   ├── 09/image_3/XXXX.png
│   └── ...
├── pointclouds/            # 点云（BIN 或自定义格式）
│   ├── 09/XXXX.bin
│   └── ...
├── imus/                   # IMU 数据（MAT 或 CSV）
│   ├── 09.mat / 09.csv
│   └── ...
├── poses/                  # 位姿（KITTI txt 或自采对齐后的文件）
│   ├── 09.txt
│   └── ...
└── meta/                   # 解析生成的索引、时间戳映射
    ├── timestamps_09.csv
    └── sensor_map.yaml
```

## 3. KITTI 解析流程
1. 执行 `data/data_prep.sh` 下载图像与位姿；补充点云下载脚本至 `pointclouds/`。
2. 根据 `timestamps.txt` 生成同步表（图像帧号 -> 时间戳）。
3. 若需要 IMU `.mat`，保持现有读取方式；如需转换 CSV，可在此阶段完成。
4. 输出：`meta/timestamps_seq.csv`、`meta/sensor_map.yaml`。

## 4. DARPA / 自采 .bag 解析
1. 使用 `rosbag` 或 Python `rosbag2` 接口，读取指定 topics：
   - 图像：`/camera/left/image_raw`、`/camera/right/image_raw`。
   - 点云：`/velodyne_points` 或 `/lidar_points`。
   - IMU：`/imu/data`。
   - 可选：`/odom`、`/tf`、`/semantic_label`。
2. 解析流程：
   - 提取消息并按时间戳保存到临时目录。
   - 将图像转换为 PNG/JPEG，点云转换为 KITTI 风格 BIN（x,y,z,intensity）。
   - IMU 保存为 CSV（time, accx, accy, accz, gyrox, gyroy, gyroz）。
   - 若存在位姿/里程计，可转换为 KITTI 轨迹格式。
3. 生成统一的时间戳映射表（图像帧、点云帧、IMU 样本的时间对齐信息）。
4. 可使用脚本模板：
   - `tools/bag2dataset.py`：输入 bag，输出统一目录结构。
   - `tools/sync_utils.py`：提供插值与同步工具。

## 5. 时间同步策略
- 图像与点云：根据时间戳匹配最近帧，保存映射表 `frame_id -> point_cloud_id`。
- IMU：使用时间窗口插值或积分，匹配到图像/点云时间。
- 若传感器启动时间不同，需设置时间偏置修正参数。

## 6. 校准与坐标系
- 存储传感器外参（例如 `calib_cam_to_velo.txt`, `calib_imu_to_cam.txt`）到 `meta/`。
- 对 DARPA、自采数据需额外标注坐标系定义，确保后续 transform 正确。
- 建议编写 `docs/data_guides/calibration.md` 记录外参格式与使用方法。

## 7. 校验与日志
- 解析脚本应输出日志（文件数、时间范围、异常条目）。
- 提供简易可视化/对齐检查（例如抽样显示图像与点云投影）。
- 构建单元测试：校验解析后目录和文件是否完整，时间表是否单调递增。

## 8. 后续任务
- 编写 `tools/bag2dataset.py`、`tools/sync_utils.py`，支持多数据源。
- 更新 README 与 `data/data_prep.sh` 说明，指导用户使用新解析流程。
- 针对 DARPA 数据收集实际样例，验证解析脚本。
