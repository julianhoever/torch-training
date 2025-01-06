# torch-training

The `torch-training` project provides predefined training loops for the PyTorch framework, simplifying the process of training classifiers. It includes support for binary and multi-class classification, with customizable training parameters and hooks.

## Installation

To install the `torch-training` package, you need to have Python 3.12 or higher. You can install the package and its dependencies using `pip`:

```bash
pip install git+https://github.com/julianhoever/torch-training@main
```

## Key Components

The key components in `src/torch_training` are:

- **`classifier/`**: Contains modules for training classifiers.
    - **`base/`**: Base components for training, including protocols, training runner, and checkpoint handler.
    - **`binary.py`**: Training loop for binary classification.
    - **`multi_class.py`**: Training loop for multi-class classification.
- **`training_hooks/`**: Contains hooks that can be used during training.
    - **`after_train_on_batch.py`**: Defines the `AfterTrainOnBatchHook` protocol that can be implemented by a model to perform an action after a single training step on a batch.
- **`history.py`**: Defines the History class for logging and saving training metrics.


## Usage

To use the predefined training loops, import the appropriate module and call the `train_model` function with your model and dataset:

```python
from torch_training.classifier.binary import train_model

history = train_model(
    model=my_model,
    ds_train=train_dataset,
    ds_val=val_dataset,
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    device="cpu"
)
```
