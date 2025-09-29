# JustAlexNet

一个基于 PyTorch 的 AlexNet 图像分类项目，涵盖数据准备、标准训练流程、极小样本场景下的迁移学习脚本以及可选的 Docker 开发环境。仓库既适合作为学习示例，也可以作为自定义实验的起点。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `alexnet.py` | 定义 AlexNet 网络结构，并提供随机张量的快速自测入口。 |
| `densenet.py` | 实现 DenseNet 主干，并提供 `densenet201` 快速构造函数。 |
| `dataset_imagenett.py` | 构建 ImageNet / ImageNet-T 目录的 `ImageFolder` 数据集，包含数据增强、训练/验证划分和 `DataLoader`。 |
| `train.py` | 主训练脚本：封装训练与验证流程、记录 Top-1/Top-5 指标、自动保存检查点并导出训练曲线。 |
| `transfer_learning.py` | 迁移学习示例：加载预训练 ResNet18，仅训练分类头，内置加权采样与早停逻辑。 |
| `inference.py` | 推理脚本：加载检查点，对单张图像输出 Top-K 预测，并支持自定义类别映射。 |
| `docker/docker-compose.yml` | 基于 NVIDIA PyTorch 镜像的 Docker Compose 环境，挂载代码与数据目录。 |
| `test.png` | 示例图片，可用于快速验证推理脚本。 |

## 环境依赖

- Python 3.10+
- PyTorch 与 Torchvision（建议使用 GPU 版本）
- 训练脚本会使用 Matplotlib 输出训练曲线
- NumPy：用于保存验证集混淆矩阵

可通过 Conda 或 `pip` 安装：

```bash
pip install torch torchvision matplotlib numpy
```

## 数据准备流程（`dataset_imagenett.py`）

1. **设置数据路径**：将 `dataset_root = Path("/app/data/ImageNet")` 修改为本地数据集目录，目录结构需满足 `split/class_name/image.jpg`。
2. **数据增强**：训练集使用缩放、随机水平翻转与 ImageNet 均值/方差归一化；验证集仅做缩放和归一化。
3. **划分比例**：默认按 80/20 划分训练与验证集，并自动为验证子集替换 transform。
4. **DataLoader 参数**：默认批大小 32、`num_workers=16`、`pin_memory=True`，可按硬件能力调整。

导入模块后可直接获取 `train_loader` 与 `val_loader`。

## 训练脚本（`train.py`）

- **设备选择**：自动检测 CUDA / Apple Silicon MPS / CPU。
- **多 GPU**：当检测到多块 CUDA GPU 时自动启用 `torch.nn.DataParallel`，日志中会输出 GPU 数量。
- **自动混合精度**：若所有 CUDA GPU 的计算能力不低于 7.0（支持 Tensor Cores），训练与验证会自动使用 `torch.cuda.amp.autocast` 与 `GradScaler`，在多 GPU 场景下获得更高吞吐并保持数值稳定。
- **优化配置**：默认使用带 Nesterov 动量的 SGD，学习率 0.01，动量 0.9，L2 权重衰减 5e-4。
- **指标记录**：每个 epoch 结束后输出训练/验证损失、Top-1/Top-5 精度，并在 `checkpoints/` 目录保存最新模型与最佳模型。
- **可视化**：训练完成后自动绘制 `training_metrics.png`，展示损失与准确率曲线。
- **验证诊断**：验证阶段会导出混淆矩阵（CSV 与热力图），帮助排查常见的分类错误模式。
- **训练恢复**：支持通过 `--resume` 指定检查点继续训练，并提供 `--early-stop-patience` 等参数控制早停逻辑。

运行方式：

```bash
python train.py --model densenet201
```


可根据需要修改 epoch 数、优化器参数或添加学习率调度器。`--model` 参数目前支持 `alexnet`、`resnet18` 与 `densenet201`。

若希望显式指定 GPU，可以使用 `CUDA_VISIBLE_DEVICES` 控制；对于需要跨节点/进程训练的场景，可参考 `torch.distributed.run` 启动方式：

```bash
python -m torch.distributed.run --nproc_per_node=<gpu_count> train.py
```

脚本会在单进程内自动落到 `DataParallel`，适合单机多卡的快速实验。

可根据需要修改 epoch 数、优化器参数或添加学习率调度器。

## 推理脚本（`inference.py`）

当模型训练完成后，可使用推理脚本对单张图像获取 Top-K 预测：

```bash
python inference.py test.png --checkpoint checkpoints/best_model.pth --model resnet18 --topk 5
```

- `--checkpoint`：指定待加载的模型权重路径。
- `--model`：选择与训练时一致的主干（`alexnet`、`resnet18` 或 `densenet201`）。
- `--topk`：设置输出的 Top-K 预测条目数，默认值为 1。
- `--class-map`：传入 `class_to_idx.json` 或其他映射文件，将预测索引还原为可读标签。
- `--device`：覆盖自动检测逻辑，例如强制在 `cpu` 或 `cuda:0` 上运行。

## 迁移学习示例（`transfer_learning.py`）

该脚本适合样本量极小的分类任务：

- 使用预训练的 ResNet18 作为骨干网络，仅训练分类头。
- 根据类频自动构建加权采样器，缓解类别不平衡。
- 支持命令行配置批大小、学习率、图像尺寸等超参数。
- 内置早停机制，并在 `artifacts/` 目录保存最佳/最终模型。

使用示例：

```bash
python transfer_learning.py /path/to/dataset \
    --batch-size 8 --epochs 25 --lr 1e-3 --workers 4
```

数据目录需符合 `ImageFolder` 结构。

## Docker 环境

`docker/docker-compose.yml` 提供了一个带 GPU 支持的开发环境：

- 基于 `nvcr.io/nvidia/pytorch` 镜像
- 将仓库挂载到容器内 `/app/code`
- 将外部数据目录挂载到 `/app/data`
- 可通过 `shm_size` 调整共享内存大小以适配 DataLoader

启动方式：

```bash
cd docker
docker compose up -d
```

## 推荐的下一步

1. **验证训练流程**：先用少量样本跑通 `python train.py`，确认损失收敛与日志/曲线输出。
2. **完善数据管道**：根据项目需求引入更贴合业务的增广、自动下载脚本或数据清洗流程。
3. **实验自动化**：结合 YAML/CLI 配置管理与日志工具（TensorBoard、Weights & Biases）批量记录实验结果。
4. **部署与集成**：将 `inference.py` 封装为 REST/gRPC 服务，或与现有系统的推理流水线对接。

通过这些步骤，可以逐步熟悉本项目的数据流、模型结构与训练流程，并在此基础上进行性能优化或功能扩展。
