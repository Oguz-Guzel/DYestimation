import math
import torch
import torch.nn as nn
import numpy as np

# Positional Encoding (optional if lepton order matters)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe.to(x.device)

class FullyConnectedNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.ReLU, hidden_layers=[32, 64]):
        super().__init__()
        layers_list = []
        input_neurons = dim_in
        for i in range(len(hidden_layers)):
            output_neurons = hidden_layers[i]
            layers_list.append(nn.Linear(input_neurons, output_neurons))
            layers_list.append(nn.LayerNorm(output_neurons))
            layers_list.append(activation())
            input_neurons = output_neurons
        layers_list.append(nn.Linear(hidden_layers[-1], dim_out))
        layers_list.append(nn.LayerNorm(dim_out))
        layers_list.append(activation())
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)

class TransformerClassifier(nn.Module):
    """Transformer encoder-only model for classification."""

    def __init__(self, input_features, num_heads, embed_dim, hidden_dim, num_classes, activation_function, num_layers=6):
        super().__init__()

        self.embeddings = nn.ModuleList()
        for feature_seq in input_features.values():
            self.embeddings.append(FullyConnectedNetwork(dim_in=len(
                feature_seq), dim_out=embed_dim, activation=activation_function))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=False)
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedding_stack = []
        for i, embedding in enumerate(self.embeddings):
            embedding_stack.append(embedding(x[i]))
        x_stack = torch.stack(embedding_stack, dim=1)
        x_transformed = self.transformer(x_stack.permute(1, 0, 2))
        x_cls = x_transformed.mean(dim=0)
        x_classifier = self.classifier(x_cls)
        output = torch.nn.functional.softmax(x_classifier, dim=-1)
        return output

    def predict(self, dataloader):
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[:-3]
                pred = self(x)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)
