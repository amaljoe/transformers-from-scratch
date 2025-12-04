import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1, num_layers=1):
        super().__init__()
        (self.input_dim, self.hidden_dim, self.output_dim, self.num_heads, self.num_layers) = (
            input_dim, hidden_dim, output_dim, num_heads, num_layers)
        self.patch_dim = int(self.hidden_dim ** 0.5)
        self.seq_len = self.input_dim * self.input_dim // (self.patch_dim * self.patch_dim)
        self.pos_emb = nn.Embedding(self.seq_len, hidden_dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.unfold(-2, self.patch_dim, self.patch_dim).unfold(-1, self.patch_dim, self.patch_dim).reshape(-1, self.seq_len, self.patch_dim * self.patch_dim)
        x = x + self.pos_emb(torch.arange(self.seq_len))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

class DigitNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = Encoder(28, 16, 10)
    x = torch.randn((28,28)).unsqueeze(0)
    out = model(x)
    print(x.shape, out.shape)
