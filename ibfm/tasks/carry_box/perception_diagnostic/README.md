# 搬箱第 0 步：感知误差鲁棒性诊断

目的：在**不重新训练**的前提下，测量冻结的 I-BFM 教师（Actor + 后向网络 B + 阶段控制器）
对机载感知误差的敏感程度，据此决定第 5 步走“估计器 + 原模型”还是“学生蒸馏”。

## 工作方式

`perception_noise.py` 作为最外层 overlay 装在 z-aware 控制器栈上（与仓库其他 overlay 相同的
monkeypatch 方式），每个控制周期（50 Hz）从真值生成**一份**感知状态，再按 `channels`
分发给三个消费方：

| 通道 | 被替换的输入 |
|---|---|
| `actor` | Actor 观测中的箱体位姿/速度、手箱接触、目标偏移（经 `ActorObservation.build` 重新生成） |
| `b` | B 的全部输入：`privileged_state`（根高度、各刚体速度、object context），以及 B 读取的 `state` 中的 object context |
| `controller` | 阶段控制器指标（`_controller_metrics`）和 `latent()` 里的箱体/目标方向 |

B 的输入过滤器是 `["state", "privileged_state"]`，和 Actor 共用 `state`；当 `actor` 与 `b` 只开其一时，
通过替换 `actor_prompt` 给 B 单独一份 `state`，携带 B 通道的 object context。

不在 `channels` 中的消费方读真值，因此可以单独扰动某一模块，也可以三者一致扰动（`all`）。
物理仿真、成功判定和 summary 中的真值结果始终使用真实 MjData。

误差注入在上游，派生通道统一重新生成：

- **箱体**：误差加在几何中心和几何坐标系上，再经 `geom_pos/geom_quat` 换回模型（标记点）坐标系；
  包含白噪声位置误差（可随距离平方增长）、时间相关误差（Ornstein-Uhlenbeck，模拟掩码截断、深度偏差这类慢变误差）、深度尺度偏差、姿态噪声、频率、延迟、丢帧、视场、
  近距离遮挡、速度来源（真值或有限差分 + 低通）、是否外推。
- **对称歧义**：`rect4 / oct8 / cube24` 三种对称群；可设初次检测随机选等价姿态、跟踪中跳变、
  遮挡后重新检测时重新选择。
- **接触**：延迟、逐拍漏检/误检。
- **机身状态估计**：根高度偏置/噪声/漂移，机身线速度偏置/噪声。感知到的箱体世界坐标同样经过
  这个估计，所以根高度误差也会让箱体高度产生同样的偏差，机身速度误差也会进入箱体速度；
  机器人相对箱体的量保持一致。
- **目标定位**：每回合偏置 + 里程计随机游走漂移。
- **控制器的高度分档**：z-aware 控制器在构建/重置时读取初始箱高（`UFO_Z3D_INITIAL_BOX_Z`）来选择低/中/高技能；
  启用 `controller` 通道时改为读取首次检测得到的感知箱高，日志中会打印真值和感知值。

另外，每拍都会在同一 latent 下用真值观测重新计算一次教师动作，记录
`||a(感知) − a(真值)||`。

## 正确性校验

- `check_perception_noise.py`：对称群、旋转映射、视图一致性、延迟/频率时序、接触延迟的自检。
  ```bash
  cd ibfm && PYTHONPATH=$PWD .venv/bin/python tasks/carry_box/perception_diagnostic/check_perception_noise.py
  ```
- `check_b_prompt_routing` 用真实权重验证：B 的 prompt 确实来自单独的 B 观测，且不会泄漏到下一次调用。
- `identity` 配置（overlay 已安装、误差为零）必须与未修改的栈**逐位一致**（已验证 all / 仅 actor / 仅 b 三种通道组合）；`run_diagnostic.py`
  在 summary 末尾报告这一项。实现要点：`mj_step` 之后 MjData 中的 xpos/cvel 等比 qpos 滞后一个
  子步，原栈直接使用它们，所以感知视图是“复制真实 MjData 再叠加误差增量”，不重新计算运动学。

## 运行

```bash
cd /home/jianhaowu/I-BFM/ibfm
.venv/bin/python tasks/carry_box/perception_diagnostic/run_diagnostic.py \
    --output-dir tasks/carry_box/perception_diagnostic/results/step0 \
    --episodes 20 --max-steps 1500 --jobs 16
```

- 默认矩阵：`baseline`、`identity/all`、11 个扰动配置 × {actor, b, controller, all}、`combined/all`。
- 所有配置使用相同的回合种子（场景布局、初始箱高、噪声流），按回合配对比较；
  `lost/gained` 是相对 baseline 由成功变失败/由失败变成功的回合数。
- 同一 `--output-dir` 再次运行会跳过已完成的回合；`--aggregate-only` 只重新汇总。
- 输出：`summary.md`、`summary.csv`（每个配置）、`episodes.csv`（每个回合）、
  `runs/<配置>/seed_<n>/`（日志、`perception_summary.json`、`height_trials.json`）。

单回合手动运行（可开 viewer 观察）：

```bash
cd ibfm/tasks/carry_box
SIM2SIM_ENTRY=$PWD/perception_diagnostic/perception_diagnostic_sim2sim.py \
UFO_PERCEPTION_PROFILE=$PWD/perception_diagnostic/profiles/combined.json \
UFO_PERCEPTION_CHANNELS=actor,b,controller \
UFO_PERCEPTION_SUMMARY_PATH=/tmp/perception_summary.json \
bash run_interactive.sh
```

`run_interactive.sh` 新增的环境变量都有默认值，不设置时行为与原来相同：`SCENE_DIR`、`OUTPUT_DIR`、
`PREPARE_SCENE`、`SIM2SIM_ENTRY`。

## 已知局限（解读结果时需注意）

1. **误差数值是占位值**。`profiles/*.json` 是根据常见传感器量级设定的起点，需要在第 1 步用真机数据标定。
   相机安装位置和俯仰角（`camera_offset_m / camera_pitch_deg`）默认取官方 URDF 的 G1 头部 D435 位姿
   （`torso_link` 上 (0.0576, 0.0175, 0.4299) m，俯角 47.6°），视场 `fov_deg` 取 D435 深度视场 87°×58°；
   这些是 CAD 设计值，仍需真机标定。
2. **丢帧时只保持或外推上一次估计**，没有用手部运动学作为先验（搬运中箱子跟着手动）。
   因此 `box_visibility` 在搬运阶段偏悲观；这本身就是第 3 步估计器需要解决的问题。
3. **动作偏差在相同 latent 下计算**。控制器只改写 object context：未改写时直接替换为真值（精确）；
   接近阶段被置零（`_masked_object_observation`）时保持置零；放置阶段的目标前视按增量叠加（近似）。
   给 B 单独一份 `state` 时规则相同。latent 选择不同带来的差异不在这个指标里，而是反映在闭环成功率中。
4. **当前场景的箱子是 0.30 m 正方体**（`run_interactive.sh` 的 `--box-size 0.30 0.30 0.30`），所以对称测试使用
   `cube24`。训练用箱子约 33.5×33.8×36.0 cm，若真机箱子能区分 z 轴，应改用 `oct8`。
5. 只覆盖搬箱；推箱、踢箱有各自的控制器快照，需要另外接入。
6. `true_pickup` 是真值定义（双手接触且箱子比初始高度高出 ≥5 cm，连续 ≥10 拍）；
   `controller_pickup` 是原控制器自己的判定（是否进入 transport），感知误差下可能误判。
