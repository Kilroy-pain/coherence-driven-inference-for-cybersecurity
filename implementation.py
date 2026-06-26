import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class WeightedGraphDataset(Dataset):
    def __init__(self, nodes, edges, weights):
        self.nodes = nodes
        self.edges = edges
        self.weights = weights

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        weight = self.weights[idx]
        return self.nodes[edge[0]], self.nodes[edge[1]], weight

class CoherenceDrivenInference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CoherenceDrivenInference, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node1, node2):
        x = torch.cat((node1, node2), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for node1, node2, weight in dataloader:
            node1, node2, weight = node1.float(), node2.float(), weight.float()
            optimizer.zero_grad()
            output = model(node1, node2).squeeze()
            loss = criterion(output, weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        predictions = []
        for node1, node2, _ in dataloader:
            node1, node2 = node1.float(), node2.float()
            output = model(node1, node2).squeeze()
            predictions.append(output)
        return torch.cat(predictions, dim=0)

if __name__ == '__main__':
    # Dummy data
    np.random.seed(42)
    torch.manual_seed(42)

    num_nodes = 10
    input_dim = 5
    hidden_dim = 16
    num_edges = 20

    nodes = torch.tensor(np.random.rand(num_nodes, input_dim), dtype=torch.float32)
    edges = torch.tensor(np.random.randint(0, num_nodes, size=(num_edges, 2)), dtype=torch.long)
    weights = torch.tensor(np.random.rand(num_edges), dtype=torch.float32)

    dataset = WeightedGraphDataset(nodes, edges, weights)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CoherenceDrivenInference(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training model...")
    train_model(model, dataloader, criterion, optimizer, epochs=10)

    print("Evaluating model...")
    predictions = evaluate_model(model, dataloader)
    print("Predictions:", predictions)