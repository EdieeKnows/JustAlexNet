# JustAlexNet

A simple PyTorch implementation of AlexNet for image classification. The training
script now also supports a ResNet-18 architecture so you can compare the two
approaches using the same training loop.

## Training

```bash
python train.py --model alexnet
python train.py --model resnet18
```

Use `--num-classes` to adapt the classifier head to your dataset.
