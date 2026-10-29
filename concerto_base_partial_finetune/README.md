# Concerto base 部分微调（concerto_base_partial_finetune）

官方 Concerto base 预训练编码器（PTv3, 108M）**部分微调**验证：解冻编码器
最后一层 stage（`enc.enc4`，约 37M 参数）+ 随机初始化二分类头。
协议与冻结实验完全一致（同数据划分、同损失、同验证集、同选模指标），
唯一区别是编码器末段参与训练。

## 在 Kaggle 上运行

1. **准备数据**：将 RandLA-Net 岩石数据上传为 Kaggle Dataset，目录结构：

   ```
   <dataset>/
     data/    1.txt ... 20.txt      (xyz+rgb，首行为 header)
     label/   1.npy ... 20.npy      (0=背景, 1=岩石, 2=ignore)
     splits/  train.txt val.txt test.txt
   ```

   将该 Dataset 附加到 Notebook（会出现在 `/kaggle/input/<slug>/`）。

2. **安装依赖**（Notebook 单元格）：

   ```bash
   pip install -r requirements.txt
   ```

   注意 `requirements.txt` 中 torch-scatter 的 `--find-links` 需匹配 Kaggle
   预装 torch 版本（先用 `python -c "import torch; print(torch.__version__, torch.version.cuda)"` 查看）。

3. **训练**：

   ```bash
   python train.py                # 正式实验：30 epochs，早停耐心 10
   python train.py --mode smoke   # 先跑 2 epoch 冒烟测试（推荐）
   ```

   - Concerto 官方源码首次运行自动下载到 `/kaggle/working/Concerto-main`
   - 预训练权重（concerto_base.pth, 414MB）自动下载到 `/kaggle/working/ckpt`
   - 法向量首次运行自动估计并缓存到 `$OUTPUT_ROOT/cache/normals`

## 输出（`/kaggle/working/outputs/concerto_base_partial_finetune_<时间戳>/`）

- `log.txt` / `history.json`：逐 epoch 训练损失 + 验证指标（采样点级与全尺寸两套）
- `confusion_ep{N}_{grid,full}.npy`：逐 epoch 混淆矩阵
- `best_head.pth` / `last_head.pth`：选模 checkpoint（验证集全尺寸岩石 IoU 最大）
- `resume.pth`：**断点续训**（模型 + 优化器 + 历史，每 epoch 原子写入）；
  重新运行 `python train.py` 自动续训（需同一输出目录，用 `--tag` 固定目录名；
  想从头开始用 `--no-resume`）
- `freeze_check.json`：冻结参数零变化验收
- `ply/`：最佳模型的预测可视化（含 FP/FN 误差图）

## 环境变量（都有默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_ROOT` | 自动探测 `/kaggle/input/*/` | 数据根目录（含 data/label/splits） |
| `OUTPUT_ROOT` | `/kaggle/working/outputs` | 日志/结果/checkpoint 输出根 |
| `CKPT_DIR` | `/kaggle/working/ckpt` | 编码器预训练权重缓存 |
| `CONCERTO_SRC` | `/kaggle/working/Concerto-main` | 官方 Concerto 源码目录 |

## 与对照实验的关系

- 受控对照（冻结探针）：本地项目 `concerto-rock-probe` 的 exp01，验证集全尺寸岩石 IoU 29.5%（epoch 11），同一数据协议
- 冻结探针基线：验证集全尺寸岩石 IoU 29.5%（exp01, epoch 11）
- 若微调后仍低于 RandLA-Net 从头训练基线（验证集岩石 IoU 45.4%），
  按 experiments.md 决策关闭预训练迁移路线
