# SkipResConnection

Pytorch implementation code for [Rethinking Skip Connections in Spiking Neural Networks]


## Abstract 
Time-To-First-Spike (TTFS) coding in Spiking Neural Networks (SNNs) offers significant advantages in terms of energy efficiency, closely mimicking the behavior of biological neurons. 
In this work, we delve into the role of skip connections, a widely used concept in Artificial Neural Networks (ANNs), within the domain of SNNs with TTFS coding. Our focus is on two distinct types of skip connection architectures: (1) addition-based skip connections, and (2) concatenation-based skip connections.
We find that addition-based skip connections introduce an additional delay in terms of spike timing. On the other hand, concatenation-based skip connections circumvent this delay but produce time gaps between after-convolution and skip connection paths, thereby restricting the effective mixing of information from these two paths. To mitigate these issues, we propose a novel approach involving a learnable delay for skip connections in the concatenation-based skip connection architecture. This approach successfully bridges the time gap between the convolutional and skip branches, facilitating improved information mixing.
We conduct experiments on public datasets inculding MNIST and Fashion-MNIST, illustrating the advantage of the skip connection in TTFS coding architectures. Additionally, we demonstrate the applicability of TTFS coding on beyond image recognition tasks and extend it to scientific machine-learning tasks, broadening the potential uses of SNNs.

## Prerequisites
* set up conda
```
conda create --name snnrescon python=3.8
conda activate snnrescon
```
* Install packages
```
pytorch 1.9.0
``` 

## Classification Experiments


```
python train_conv_snndirect.py --dataset fmnist --arch shuffle
```

## Classification Experiments

```
python train_conv_snndirect_wave.py --arch shuffle
```

## About Quantization-Aware Training (QAT)
We also provide ``quantization-aware training``.

To enable quantization, set ``QUANTIZATION_FLAG`` to ``True``.



## Acknowledgement 
This code is based on Tensorflow code: https://github.com/zbs881314/Temporal-Coded-Deep-SNN
