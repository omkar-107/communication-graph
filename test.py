import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
import networkx as nx
import matplotlib.pyplot as plt
from faker import Faker
import random
import numpy as np


# Enhanced Graph Schema and Data Generation
class AdvancedCommunicationGraph:
    def __init__(self, num_nodes=1000, avg_degree=5):
        self.fake = Faker()
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.node_features = []
        self.edge_features = []
        self.edge_index = []
        
        # Security clearance mapping
        self.clearance_levels = {
            'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4
        }
        
    def generate_nodes(self):
        """Generate nodes with enhanced attributes"""
        for _ in range(self.num_nodes):
            node = {
                'email': self.fake.unique.email(),
                'department': random.choice(['ENG', 'HR', 'FIN', 'IT', 'SALES']),
                'clearance': random.choice(list(self.clearance_levels.keys())),
                'risk_score': np.random.beta(1, 5),  # Most users low risk
                'failed_logins': np.random.poisson(0.3),
                'access_level': random.randint(1, 5),
                'tenure': np.random.lognormal(2, 0.5)
            }
            self.node_features.append(node)
        return self
    
    def generate_edges(self):
        """Generate edges with contextual relationships"""
        G = nx.barabasi_albert_graph(self.num_nodes, self.avg_degree//2)
        edge_indices = list(G.edges())
        
        for u, v in edge_indices:
            src = self.node_features[u]
            dst = self.node_features[v]
            
            edge = {
                'src': u,
                'dst': v,
                'comms': np.random.poisson(15),
                'encrypted': random.choices([0, 1], weights=[0.7, 0.3])[0],
                'sensitive_keywords': np.random.poisson(0.5),
                'attachments': np.random.poisson(2),
                'time_diff_mean': abs(np.random.normal(0, 2)),
                'clearance_diff': abs(self.clearance_levels[src['clearance']] - 
                                    self.clearance_levels[dst['clearance']])
            }
            
            # Add inverse edge for undirected graph
            self.edge_features.append(edge)
            self.edge_index.append([u, v])
            self.edge_index.append([v, u])
            
        return self
    
    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object"""
        node_df = pd.DataFrame(self.node_features)
        edge_df = pd.DataFrame(self.edge_features)
        
        # Normalize features
        scaler = StandardScaler()
        node_feats = scaler.fit_transform(pd.get_dummies(
            node_df[['department', 'clearance', 'risk_score', 
                    'failed_logins', 'access_level', 'tenure']]
        ))
        
        edge_feats = scaler.fit_transform(
            edge_df[['comms', 'encrypted', 'sensitive_keywords', 
                    'attachments', 'time_diff_mean', 'clearance_diff']]
        )
        
        return Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_feats, dtype=torch.float)
        )

# Graph Anomaly Detection Model
class GraphAnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.encoder = None
        self.decoder = None
        self.edge_scores = None
        
    def train_gae(self, epochs=100):
        """Train custom GAE for edge feature reconstruction"""
        class Encoder(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, 2*out_channels)
                self.conv2 = GCNConv(2*out_channels, out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                return self.conv2(x, edge_index)
            
        class Decoder(torch.nn.Module):
            def __init__(self, in_channels, edge_feat_dim):
                super().__init__()
                self.linear = torch.nn.Linear(2 * in_channels, edge_feat_dim)

            def forward(self, z, edge_index):
                src, dst = edge_index
                edge_emb = torch.cat([z[src], z[dst]], dim=1)
                return self.linear(edge_emb)
        
        in_channels = self.data.num_node_features
        out_channels = 16
        edge_feat_dim = self.data.edge_attr.size(1)

        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, edge_feat_dim)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()), 
            lr=0.01
        )
        
        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = self.encoder(self.data.x, self.data.edge_index)
            recon = self.decoder(z, self.data.edge_index)
            loss = torch.nn.functional.mse_loss(recon, self.data.edge_attr)
            loss.backward()
            optimizer.step()
            
        # Calculate reconstruction error per edge
        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)
            recon = self.decoder(z, self.data.edge_index)
            self.edge_scores = (recon - self.data.edge_attr).pow(2).mean(dim=1).numpy()
            
        return self
    
    def detect_structural_anomalies(self):
        """Detect node-level anomalies using ECOD"""
        detector = ECOD()
        detector.fit(self.data.x.numpy())
        return detector.decision_scores_
    
    def detect_edge_anomalies(self):
        """Combine reconstruction error with edge features using Isolation Forest"""
        edge_features = np.column_stack([
            self.edge_scores,
            self.data.edge_attr.numpy()
        ])
        
        detector = IForest(contamination=0.05)
        detector.fit(edge_features)
        return detector.decision_scores_
    
    def visualize_anomalies(self, top_k=20):
        """Visualize most anomalous connections"""
        G = nx.Graph()
        edge_scores = self.detect_edge_anomalies()
        top_edges = sorted(zip(self.data.edge_index.t().numpy(), edge_scores),
                          key=lambda x: -x[1])[:top_k]
        
        plt.figure(figsize=(12, 8))
        for (u, v), score in top_edges:
            G.add_edge(u, v, weight=score)
            
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_edges(G, pos, edge_color='r', alpha=0.5,
                              width=[G[u][v]['weight']*2 for u,v in G.edges()])
        plt.title("Top Anomalous Connections")
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Generate enhanced graph data
    graph = AdvancedCommunicationGraph(num_nodes=500, avg_degree=8)
    graph.generate_nodes().generate_edges()
    pyg_data = graph.to_pyg_data()
    
    # Initialize and train detector
    detector = GraphAnomalyDetector(pyg_data)
    detector.train_gae(epochs=50)
    
    # Detect anomalies
    node_anomalies = detector.detect_structural_anomalies()
    edge_anomalies = detector.detect_edge_anomalies()
    
    # Visualize results
    print(f"Node anomaly scores range: {node_anomalies.min():.2f} - {node_anomalies.max():.2f}")
    print(f"Edge anomaly scores range: {edge_anomalies.min():.2f} - {edge_anomalies.max():.2f}")
    
    detector.visualize_anomalies()