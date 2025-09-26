# JustAlexNet

一个基于 PyTorch 的 AlexNet 图像分类模型实现，涵盖数据准备、训练脚本以及可选的 Docker 开发环境配置。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `alexnet.py` | 定义 AlexNet 网络结构，并在模块末尾提供一个使用随机张量的快速自测。 |
| `dataset_imagenett.py` | 使用 torchvision 的 `ImageFolder` 封装数据集，包含数据增强、标准化以及训练 / 验证数据加载器。 |
| `train.py` | 整合模型与数据加载器，配置优化器与损失函数，并运行训练循环。 |
| `docker/docker-compose.yml` | 提供基于 NVIDIA PyTorch 镜像的 Docker Compose 环境，可挂载代码与数据目录。 |
| `test.png` | 示例图片，可用于快速验证推理脚本。 |

项目依赖主要集中在 PyTorch 与 Torchvision。数据与模型权重默认存放在仓库外部，便于管理大文件。

## 数据准备流程（`dataset_imagenett.py`）

1. **数据集路径**：脚本默认读取 `dataset_root = Path("/app/data/ImageNet")`。请将其修改为本地 ImageNet 或 ImageNet-T 数据所在目录，并确保遵循 `split/class_name/image.jpg` 的层级结构。
2. **数据增强**：训练集会被缩放到 224×224，随机水平翻转并使用 ImageNet 均值 / 方差归一化；验证集仅进行缩放与归一化。
3. **训练 / 验证划分**：默认按照 80/20 划分数据集，划分后会自动为验证子集替换成对应的 transform。
4. **DataLoader 参数**：默认批大小为 32、`num_workers=16`，并开启 `pin_memory=True`，可根据硬件资源调整。

导入该模块后即可获得 `train_loader` 与 `val_loader` 供训练脚本直接使用。

## 模型与训练流程（`alexnet.py`, `train.py`）

- **设备选择**：自动检测 CUDA、Apple Silicon 的 MPS 或 CPU，保证脚本在不同平台上运行。
- **模型定义**：网络结构与经典 AlexNet 一致，包含特征提取的卷积层与分类器的全连接层。
- **训练循环**：`train_loop` 会遍历批次、计算交叉熵损失、反向传播并每 100 个 batch 打印日志。`train.py` 默认使用带动量与权重衰减的 SGD，可选启用 Nesterov。
- **运行方式**：执行 `python train.py` 会初始化数据加载器、模型、损失函数与优化器，并运行 10 个 epoch。可在此基础上扩展保存权重、指标评估等功能。

## Docker 开发环境（`docker/docker-compose.yml`）

Compose 文件会启动 NVIDIA 官方 PyTorch 镜像，自动挂载仓库到容器内 `/app/code`，并将外部数据目录挂载到 `/app/data`。根据 DataLoader 需求可以调整共享内存大小 (`shm_size`)。

## 新人学习建议

1. **跑通基线**：选取少量样本运行 `python train.py`，确认训练流程正常，观察损失曲线是否收敛。
2. **补充验证指标**：在 `train.py` 中添加使用 `val_loader` 的评估流程，记录 top-1 精度或损失。
3. **实现断点续训**：定期调用 `torch.save(model.state_dict(), ...)` 保存权重，并编写加载逻辑。
4. **调参实验**：尝试不同的批大小、学习率、学习率调度器（如 `StepLR`、`CosineAnnealingLR`）或优化器（Adam、SGD + Cosine）。
5. **配置管理**：将路径、批大小、训练轮数等超参数抽离到配置文件或命令行参数，方便复现。
6. **扩展推理脚本**：编写独立的推理脚本，加载训练好的模型对新图片（如 `test.png`）进行分类。

通过上述步骤，您可以逐步理解本仓库的数据流、模型结构与训练逻辑，并在此基础上继续迭代功能或性能。
