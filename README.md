<p align="center">
  <a href="assets/real-robot/g1-carry-box.mp4">
    <img src="assets/real-robot/g1-carry-box.gif" alt="G1 真机搬箱" width="820">
  </a>
</p>

<h1 align="center">I-BFM</h1>

<p align="center">
  G1 走到纸箱前，把它抱起来，再抱着穿过实验室。<br>
  搬、推、踢用的是同一个策略。
</p>

<p align="center">
  <a href="assets/real-robot/g1-carry-box.mp4">原视频</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2610.06129">论文</a>
  &nbsp;·&nbsp;
  <a href="https://iamhardworking.github.io/I-BFM/">项目网站</a>
  &nbsp;·&nbsp;
  <a href="README.en.md">English</a>
</p>

---

给策略一个任务奖励，它自己选出该怎么动。没有为搬箱、推箱、踢箱各训一份网络。跌倒或被推开之后，它会接着把箱子送到该去的地方。

真机片段大约 39 秒。画面点进去是带声音的原文件。

## 仿真

<table>
  <tr>
    <td align="center" width="33%"><b>搬</b></td>
    <td align="center" width="33%"><b>推</b></td>
    <td align="center" width="33%"><b>踢</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/simulations/carry-box.gif" alt="仿真搬箱" width="100%"></td>
    <td align="center"><img src="assets/simulations/push-box.gif" alt="仿真推箱" width="100%"></td>
    <td align="center"><img src="assets/simulations/kick-box.gif" alt="仿真踢箱" width="100%"></td>
  </tr>
</table>

三份脚本都在 `ibfm/` 里，读的是同一次训练留下的 166.560M 步检查点。

```bash
./tasks/carry_box/run_interactive.sh
./tasks/push_box/run_interactive.sh
./tasks/kick_box/run_interactive.sh
```

## 在这台机器上跑

需要 Linux、Python 3.10、`uv` 和 MuJoCo。这个仓库只放代码和视频，模型权重没有上传。

```bash
git clone https://github.com/HappyBigShark/I-BFM.git
cd I-BFM/ibfm
uv sync
```

八张 H200 上重训：

```bash
./scripts/train_ibfm_h200.sh ibfm_reproduction
```

中断后用 `RESUME=1` 接着跑。没有同一时刻的 replay buffer 时，续训结果会和一次跑完不一样。

166.560M 步的 replay buffer 在 [Hugging Face](https://huggingface.co/datasets/HanSj77/I-BFM-166560M-replay-buffer)，八个分片，大约 210 GiB。

## 代码在哪

| 路径 | 内容 |
| --- | --- |
| `ibfm/tasks/carry_box/` | 搬箱 |
| `ibfm/tasks/push_box/` | 推箱 |
| `ibfm/tasks/kick_box/` | 踢箱 |
| `ibfm/tasks/carry_box/vision_distillation/` | 只用机载相机的搬箱学生，权重未附 |
| `assets/real-robot/` | 上面这段真机视频 |
| `checkpoints/ibfm-166m/` | 检查点说明。权重文件不在这个仓库里 |

许可是 [CC BY-NC 4.0](LICENSE)。

```bibtex
@misc{han2026ibfm,
  title         = {{I-BFM}: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning},
  author        = {Ziqi Han and Yitang Li and Junhan Sun and Fanrong Dong and Yaojie Shen and Lei Ye and Zetong Jing and Yongqi Zhang and Yiming Zhang and Xue Wang and Hao Zhao},
  year          = {2026},
  eprint        = {2610.06129},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.06129}
}
```
