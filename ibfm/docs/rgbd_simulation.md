# 仿真 RGB-D 相机

提供机身固定的 RGB、米制深度和可见箱子掩码，不需要连接真机或下载新的权重。
相机安装在 G1 的 `torso_link` 上，随身体（含腰关节）运动；不是现有第三人称观察相机。
默认配置对应 G1 头部的 RealSense D435i，位姿取自 Unitree 官方
`g1_description`（rev_1_0）URDF 的 `d435_joint`：

| 参数 | 值 | 来源 |
|---|---|---|
| 安装刚体 | `torso_link` | URDF `d435_joint` 父刚体 |
| 位置 | `(0.0576235, 0.01753, 0.42987)` m | URDF `xyz` |
| 俯角 | 47.6°（0.8308 rad） | URDF `rpy` 的 pitch |
| 垂直视场 | 58° | D435 深度视场 87°×58° |
| 分辨率 | 848×480 | D435 推荐深度分辨率 |

该 frame 是深度原点（左 IR 相机），所以有 1.75 cm 横向偏移；RGB 与深度共用这个视点，
相当于已对齐到深度的彩色图，不是独立的 69°×42° 彩色相机。这是 CAD 设计位姿，
不是单台机器人的标定；有实测内外参后应替换。

## 运行采集示例

从仓库根目录运行：

```bash
cd ibfm
MUJOCO_GL=egl .venv/bin/python -m humanoidverse.tools.capture_rgbd
```

默认使用现有 G1 模型、默认关节姿态和前方 1.0 m 处地面上的 0.30 m 正方体箱子，采集一个静态帧，
保存到新建的 `runs/rgbd_<时间>/`。这是相机测试，不执行搬箱策略或物理控制流程。
无窗口渲染需要可用的 EGL/OpenGL 环境；不需要加载策略 checkpoint。

可以覆盖分辨率、俯角、视场和箱子位置（例如模拟 D435 彩色相机的 42° 垂直视场）：

```bash
MUJOCO_GL=egl .venv/bin/python -m humanoidverse.tools.capture_rgbd \
  --width 640 --height 360 --fovy 42 \
  --box-pos 0.5 0.0 0.75 --output-dir runs/rgbd_near_box
```

`--output-dir` 必须是尚不存在的目录，避免覆盖旧数据。
`--camera-pos X Y Z` 是相对安装刚体的位置（米）；`--box-pos` 是箱子中心的世界坐标（米）。
未指定时使用上表的官方 D435 配置，光轴沿安装刚体的 +X 方向向下俯仰 47.6°。

## 输出与读取

- `rgb.png`：RGB 预览。
- `depth_preview.png`：0–5 m 映射成灰度的预览，不能用于几何计算。
- `box_mask.png`：可见箱子像素，遮挡部分不会补全。
- `frame.npz`：原始数据、相机内外参、仿真时间戳和箱子可见点云。
- `metadata.json`：配置和单位说明。

```python
import numpy as np
from humanoidverse.envs.rgbd_camera import depth_to_points

with np.load("runs/rgbd_near_box/frame.npz", allow_pickle=False) as frame:
    rgb = frame["rgb"]                  # uint8 [H, W, 3]
    depth = frame["depth_m"]            # float32 [H, W]，单位米
    valid = frame["depth_valid"]        # bool [H, W]，排除背景/裁剪面
    box_mask = frame["box_mask"]        # bool [H, W]，仿真分割真值
    points = depth_to_points(depth, frame["K"], valid & box_mask)
    transform = frame["T_base_camera"]
    points_base = points @ transform[:3, :3].T + transform[:3, 3]
```

深度是光学坐标系的 Z，不是到相机的欧氏距离。光学坐标轴为右、下、前。
`K` 使用整数像素索引为像素中心的约定；`T_world_camera` / `T_base_camera`
分别将光学坐标转换到世界 / 机器人基座刚体坐标。默认基座为 `pelvis`。
世界变换是仿真真值，仅供校验，不表示真实部署时可获得世界定位。

## 接入控制循环

`humanoidverse.envs.rgbd_camera` 暴露：

- `RgbdCameraConfig`：分辨率、安装刚体、位置、俯角、垂直视场。
- `add_rgbd_camera(spec, config)`：在 `MjSpec.compile()` 前添加相机。
- `RgbdCamera(model, config, box_geom=...)`：创建离屏渲染器。
- `camera.capture(data)`：获取同一仿真状态下的 RGB、深度、掩码及标定数据。

现有 `carry_goal_sim2sim._build_model` 也支持可选参数 `rgbd_camera=config`。
修改 `qpos` 或完成物理步进后，调用 `mujoco.mj_forward(model, data)` 更新运动学，
再调用 `capture`。采集的三个渲染通道之间不会推进物理时间。使用上下文管理器
或显式调用 `camera.close()` 释放渲染器；不要在其他线程并发修改 `data`。

当前没有给三个 `run_interactive.sh` 添加采集开关，也没有修改训练环境或策略观测。
原 Actor/B 仍使用状态量：**生成 RGB-D 不等于策略已经会依据图像控制**。
本接口是后续感知估计、视觉学生训练的数据入口；箱体拟合、噪声/延迟模拟、
阶段控制器视觉切换、学生蒸馏和 DAgger 尚未实现。

## 测试

```bash
cd ibfm  # 如果已在 ibfm 目录中，则省略
MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -p test_rgbd_camera.py -v
```

测试覆盖米制深度、内外参/像素投影、遮挡掩码、相机跟随机身、输入校验和帧数据独立性。
