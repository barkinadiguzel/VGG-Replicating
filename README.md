# 🖋 VGG16-PyTorch-Implementation

This repository contains a PyTorch implementation of **VGG16**, reproducing the architecture from the paper **Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015)**.  

- Implements **VGG16 configuration** with stacked 3×3 convolutional blocks, ReLU activations, max-pooling, flattening, and fully connected layers (4096 → 4096 → 1000).  
- Designed for ImageNet classification while allowing easy extension to other VGG variants (A–E) or custom networks.  

**Paper**: [VGG16 Paper (ICLR 2015)](https://arxiv.org/abs/1409.1556)

---

## 🖼 Overview – VGG16 Architecture

- **Convolutional Blocks:** Stacked 3×3 convolutions with ReLU activations.  
- **Max-Pooling Layers:** Reduce spatial dimensions, provide translation invariance.  
- **Flatten Layer:** Converts convolutional feature maps to a vector for fully connected layers.  
- **Fully Connected Layers:** FC6 → FC7 → FC8 (4096 → 4096 → 1000) with dropout (0.5) on first two layers.  
- **Input/Output:** 224×224×3 images, output 1000-class softmax probabilities.  
- **Key Idea:** Deeper configurations with small stacked filters (D, E) achieve better accuracy than shallower variants (A–C).  

![VGG16 Overview](images/figmix.jpg)  
*Figure:* Unified visual of VGG16 architecture combining convolutional, pooling, and fully connected layers.

---

## 🔑 Key Formulas – VGG16

1. **Convolutional Layer:**  

$$
y_{i,j,k}^{(l)} = f\Bigg(\sum_{c=1}^{C_{l-1}} \big(x^{(l-1)}_c * W^{(l)}_{k,c}\big)_{i,j} + b^{(l)}_k\Bigg)
$$

- $x^{(l-1)}_c$ = input feature map of previous layer  
- $W^{(l)}_{k,c}$ = convolution kernel  
- $b^{(l)}_k$ = bias  
- $f$ = ReLU activation

2. **Fully Connected Layer:**  

$$
y = f(Wx + b)
$$

- Flattens convolutional output  
- Maps to output classes with softmax at final layer  
- Optional dropout applied to first two FC layers

> These formulas summarize **VGG16’s core computations**: hierarchical feature extraction and end-to-end classification learning.

---

## 🏗 Project Structure

```bash
VGG16-PyTorch-Implementation/
│
├── src/
│   ├── layers/
│   │   ├── conv_block.py       # 3x3 Conv + ReLU + N repeats
│   │   ├── maxpool_layer.py    # MaxPool2d layer
│   │   ├── flatten_layer.py    # Flatten (Conv→FC transition)
│   │   └── fc_layer.py         # Fully Connected Layer + ReLU/Dropout
│   │
│   ├── model/
│   │   └── vgg.py              # Conv Blocks + MaxPool + Flatten + FC Layers Assembly
│   │
│   ├── pretrain.py             # Training loop
│   └── config.py               # Hyperparameters, dataset path, optimizer
│
├── images/                     
│   └── figmix.jpg
└── requirements.txt

```

---

## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)


