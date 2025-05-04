# FashionMNIST Classification with a Neural Network

## Overview

Repo for practicing fully connected neural network classification model.


## Repository Structure
```
FashionMNIST-Classification/
│── Notes/                  # Additional notes
│── FashionMNIST.py         # Main script containing model training and evaluation
│── README.md               # Project documentation
```

## Model Architecture
```python
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

## Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 70%    |
| Avg Loss  | 0.789  |

## Training Parameters
- Loss Function    |   CrossEntropyLoss
- Optimizer        |   Stochastic Gradient Descent (SGD)
- Batch Size       |   64
- Learning Rate    |   0.001
- Epochs           |   10

## References
This model is based on [this PyTorch tutorial](https://pytorch.org/tutorials/).

## License
This project is open-source. Feel free to use and modify it.