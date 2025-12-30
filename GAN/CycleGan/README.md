# CycleGAN: Summer-to-Winter Image Translation

This repository contains a PyTorch implementation of **CycleGAN** (Cycle-Consistent Adversarial Networks) for unpaired image-to-image translation. The model is trained to translate images of **Yosemite National Park from Summer to Winter** (and vice versa) without requiring paired training data.

![Sample Output](sample_32600.jpg)
*Row 1: Real Summer | Row 2: Generated Winter | Row 3: Real Winter | Row 4: Generated Summer*

## 📝 Project Overview
CycleGAN learns a mapping between two domains (X and Y) by using two generators and two discriminators. The key innovation is the **Cycle Consistency Loss**, which ensures that if we translate an image from Summer → Winter and back to Summer, we recover the original image ($F(G(X)) \approx X$).

### Key Features
* **Architecture:** * **Generators:** ResNet-based architecture with 9 residual blocks for high-quality style transfer.
    * **Discriminators:** 70x70 PatchGAN (focuses on local texture details rather than global structure).
* **Stabilization:** Uses an Image Replay Buffer to stabilize Discriminator training.
* **Checkpointing:** Includes logic to automatically save and resume training from Google Drive (Colab compatible).

## 📊 Results (Epoch 105/200)
The model is currently trained for 105 epochs on the `summer2winter_yosemite` dataset.
- **Summer → Winter:** Successfully adds snow textures to trees and cools down the color temperature.
- **Winter → Summer:** Removes snow coverage and hallucinates greenery in appropriate areas.

## 🛠️ Requirements
* Python 3.8+
* PyTorch
* Torchvision
* Matplotlib
* Pillow

```bash
pip install torch torchvision matplotlib pillow
