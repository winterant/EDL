Classification Uncertainty
---

## 🧐 About

An easy-to-run implementation using PyTorch for the paper
"
[*Evidential Deep Learning to Quantify Classification Uncertainty*](http://arxiv.org/abs/1806.01768)
".

## 🎈 Usage

**Requirements**

+ python 3.8.8
+ numpy 1.19.2
+ pytorch 1.7.1
+ torchvision 0.8.2
+ scikit-learn 0.24.1
+ scikit-image 0.18.2
+ scipy 1.6.2

**Running**

```shell
python main.py
```

## 📝 Experiments

Acc on valid dataset.

<div align="center">

| Model | MNIST |
|---|---|
|softmax|0.9909|
|EDL using log|0.9714|
|EDL using digamma|0.9766|
|EDL using mse|0.9745|

</div>

## :gift_heart: Acknowledgement

+ [atilberk/evidential-deep-learning-to-quantify-classification-uncertainty](https://github.com/atilberk/evidential-deep-learning-to-quantify-classification-uncertainty)

+ [dougbrion/pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty)
