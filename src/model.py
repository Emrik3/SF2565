"""
Configurable Neural Network for pre-extracted features.
Supports multiple architectures and easy parameter tuning.
"""

import torch
import torch.nn as nn


class FeatureNN(nn.Module):
    """
    Flexible Neural Network for classification on pre-extracted features.

    Args:
        input_dim: Dimension of input features (e.g., 2048 for ResNet50)
        hidden_dims: List of hidden layer dimensions. E.g., [512, 256] creates 2 hidden layers
        num_classes: Number of output classes (1 for binary classification)
        dropout_rate: Dropout probability (0.0 to 1.0)
        activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
        batch_norm: Whether to use batch normalization
        use_residual: Whether to add residual connections (requires compatible dimensions)
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dims=[512, 256],
        num_classes=1,
        dropout_rate=0.5,
        activation="relu",
        batch_norm=True,
        use_residual=False,
    ):
        super(FeatureNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        # Build activation function
        self.activation_fn = self._get_activation(activation)

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self._get_activation(activation))

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation):
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(activations.keys())}"
            )
        return activations[activation.lower()]

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        x = self.feature_layers(x)
        x = self.output_layer(x)
        return x

    def get_num_parameters(self):
        """Returns the total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self):
        """Returns information about each layer"""
        info = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.Dropout)):
                info.append(f"{name}: {module}")
        return "\n".join(info)


class SimpleFeatureNN(nn.Module):
    """
    Simple 2-layer neural network for quick testing.
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=1, dropout_rate=0.5):
        super(SimpleFeatureNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepFeatureNN(nn.Module):
    """
    Deeper network with residual connections for better gradient flow.
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dims=[1024, 512, 256, 128],
        num_classes=1,
        dropout_rate=0.3,
    ):
        super(DeepFeatureNN, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                self._make_residual_block(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate
                )
            )

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], num_classes)

    def _make_residual_block(self, in_dim, out_dim, dropout_rate):
        """Creates a residual block"""
        return nn.ModuleDict(
            {
                "linear": nn.Linear(in_dim, out_dim),
                "bn": nn.BatchNorm1d(out_dim),
                "relu": nn.ReLU(inplace=True),
                "dropout": nn.Dropout(dropout_rate),
                "skip": nn.Linear(in_dim, out_dim)
                if in_dim != out_dim
                else nn.Identity(),
            }
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = nn.functional.relu(x)

        # Hidden layers with residual connections
        for block in self.layers:
            identity = block["skip"](x)

            out = block["linear"](x)
            out = block["bn"](out)
            out = block["relu"](out)
            out = block["dropout"](out)

            x = out + identity

        # Output
        x = self.output(x)
        return x


def test_models():
    """Test all model architectures"""
    batch_size = 32
    input_dim = 2048

    print("=" * 60)
    print("Testing FeatureNN (Flexible)")
    print("=" * 60)
    model1 = FeatureNN(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        num_classes=1,
        dropout_rate=0.5,
        activation="relu",
        batch_norm=True,
    )
    x = torch.randn(batch_size, input_dim)
    y = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {model1.get_num_parameters():,}")
    print()

    print("=" * 60)
    print("Testing SimpleFeatureNN")
    print("=" * 60)
    model2 = SimpleFeatureNN(input_dim=input_dim, hidden_dim=512, num_classes=1)
    y = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print()

    print("=" * 60)
    print("Testing DeepFeatureNN")
    print("=" * 60)
    model3 = DeepFeatureNN(
        input_dim=input_dim,
        hidden_dims=[1024, 512, 256, 128],
        num_classes=1,
        dropout_rate=0.3,
    )
    y = model3(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model3.parameters()):,}")
    print()


if __name__ == "__main__":
    test_models()
