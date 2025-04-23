# ANN-PyTorch: Simple Neural Network on the Iris Dataset

This project features a simple 3-layer fully connected (ANN) neural network built using PyTorch and trained on the Iris dataset. It serves as a solid starting point for those looking to understand basic neural network implementation and classification tasks.

## Project Structure

- **`model.py`**: Defines the architecture of the neural network.
- **`train.py`**: Trains the model. Includes hyperparameter settings and the training loop.
- **`test.py`**: Tests the trained model with new input data.

## Model Details

- **Input Layer**: 4-dimensional vector (sepal/petal length and width)
- **Hidden Layers**: Fully connected (dense) layers
- **Output Layer**: 3 classes (Setosa, Versicolor, Virginica)
- **Activation Function**: `ReLU`
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam`

## Training

```bash
python train.py
