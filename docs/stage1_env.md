# 环境依赖与脚本整理

## 1. 开发环境
- 操作系统：Ubuntu 20.04+/WSL2（推荐），兼容 Windows + WSL。
- Python：3.10（与现有环境一致）。
- CUDA：12.x（根据 GPU 驱动调整）。
- GPU：单卡 RTX 4060（8GB）作为基准，注意模型规模。

## 2. Python 依赖
```bash
conda create -n botvio python=3.10
conda activate botvio

# PyTorch + CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 基础依赖
pip install timm==0.4.12 einops gdown matplotlib scipy scikit-image opencv-python tqdm

# 点云/语义后续依赖（根据选型补充）：
pip install open3d MinkowskiEngine torch-scatter torchnet # 根据模型选择
pip install mmsegmentation mmcv-full                      # 图像语义（可选）
pip install opencv-contrib-python                         # 图像增强（可选）
```
- 根据实际选定的点云/语义模型，动态维护 `requirements.txt`。

## 3. C++/系统依赖
- `build-essential`, `cmake`, `libeigen3-dev`, `libboost-all-dev`（如需编译点云库）。
- `ros-noetic-desktop` 或 `ros2-foxy`（若要解析 bag 并做实时测试）。
- `libpcl-dev`（如果在 C++ 层处理点云）。

## 4. 数据脚本
- `data/data_prep.sh`：保留原功能，添加点云下载（KITTI velodyne）。
- 新增：
  - `tools/bag2dataset.py`：解析 ROS bag，输出统一目录结构。
  - `tools/sync_utils.py`：时间同步、插值工具。
  - `tools/check_dataset.py`：检查数据完整性、生成报告。

## 5. 测试脚本
- `tests/test_data_pipeline.py`：验证解析后目录结构、时间戳文件。
- `tests/test_forward.py`：加载模型权重，跑单个 batch 前向，确保无 NaN。
- `tests/test_semantics.py`：语义模块输出检查（类别数量、置信度范围等）。

## 6. 文档更新计划
- README：补充环境安装、数据解析、三模态说明。
- docs/：
  - `data_guides/` 目录用于记录不同数据集解析方法。
  - `models/` 目录用于记录编码器/语义模型配置。

## 7. 偏差处理
- 若后续模块引入新依赖，需在此文档追加记录并同步到 README。
- 任何与 GPU/驱动相关的问题需有 troubleshooting 章节，记录解决方案。
