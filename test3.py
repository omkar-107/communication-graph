import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

class TemporalCommunicationGraph:
    def __init__(self, num_nodes=1000, avg_degree=8, time_windows=24):
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.time_windows = time_windows
        self.nodes = []
        self.temporal_edges = defaultdict(list)
        self.clearance_levels = ['L1', 'L2', 'L3', 'L4']
        self.departments = ['ENG', 'HR', 'FIN', 'IT', 'SALES']
        self.base_time = datetime.now()
        self.base_edges = None

    def generate_nodes(self):
        for _ in range(self.num_nodes):
            self.nodes.append({
                'id': _,
                'department': np.random.choice(self.departments),
                'clearance': np.random.choice(self.clearance_levels),
                'risk_score': np.random.beta(1, 5),
                'login_attempts': np.random.poisson(3),
                'access_level': np.random.randint(1, 5),
                'tenure': np.abs(np.random.normal(2, 0.5)),
                'auth_failures': np.random.poisson(0.5)
            })
        return self

    def generate_temporal_edges(self):
        # Generate fixed base graph structure
        G = nx.barabasi_albert_graph(self.num_nodes, self.avg_degree//2)
        self.base_edges = list(G.edges())
        
        # Use the same edge set for all time windows
        for t in range(self.time_windows):
            current_time = self.base_time + timedelta(hours=t)
            
            for u, v in self.base_edges:
                # Add time-varying features while keeping edge structure constant
                self.temporal_edges[t].append({
                    'source': u,
                    'target': v,
                    'timestamp': current_time,
                    'comms_volume': np.random.poisson(20 * (1 + np.sin(t/24 * 2 * np.pi))),
                    'encrypted_ratio': np.random.beta(2, 5),
                    'sensitive_keywords': np.random.poisson(1.5),
                    'time_diff': np.abs(np.random.normal(0, 3)),
                    'clearance_diff': abs(self.clearance_levels.index(self.nodes[u]['clearance']) - 
                                        self.clearance_levels.index(self.nodes[v]['clearance'])),
                    'hour_of_day': current_time.hour,
                    'is_weekend': current_time.weekday() >= 5,
                    'temporal_pattern': np.sin(t/24 * 2 * np.pi)
                })
        return self

    def get_time_window_data(self, time_idx):
        node_df = pd.DataFrame(self.nodes)
        categorical_features = ['department', 'clearance']
        numeric_features = ['risk_score', 'login_attempts', 'access_level', 'tenure', 'auth_failures']
        
        node_df = pd.get_dummies(node_df, columns=categorical_features)
        node_features = node_df[numeric_features + 
                              [c for c in node_df.columns if c.startswith('department_') or c.startswith('clearance_')]]
        
        edge_df = pd.DataFrame(self.temporal_edges[time_idx])
        edge_index = torch.tensor([edge_df['source'], edge_df['target']], dtype=torch.long)
        
        edge_features = edge_df[[
            'comms_volume', 'encrypted_ratio', 'sensitive_keywords', 
            'time_diff', 'clearance_diff', 'hour_of_day', 
            'temporal_pattern'
        ]]
        
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        edge_features = scaler.fit_transform(edge_features)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            time_idx=torch.tensor([time_idx], dtype=torch.long)
        )

class TemporalGNNAnomalyDetector(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64):
        super().__init__()
        self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_dim)
        self.temporal_attention = torch.nn.Linear(hidden_dim, 1)
        self.conv1 = GATConv(num_node_features, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data, temporal_memory=None):
        edge_emb = self.edge_encoder(data.edge_attr)
        
        x = F.relu(self.conv1(data.x, data.edge_index, edge_emb))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index, edge_emb)
        
        if temporal_memory is None:
            temporal_memory = torch.zeros_like(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x.unsqueeze(0), None)
        temporal_features = lstm_out.squeeze(0)
        
        row, col = data.edge_index
        edge_features = torch.cat([
            x[row], 
            x[col], 
            temporal_features[row]
        ], dim=1)
        
        return torch.sigmoid(self.scorer(edge_features)).squeeze(), (h_n, c_n)

class TemporalAnomalyDetectionFramework:
    def __init__(self, temporal_graph):
        self.temporal_graph = temporal_graph
        self.time_windows = temporal_graph.time_windows
        
        sample_data = temporal_graph.get_time_window_data(0)
        
        self.model = TemporalGNNAnomalyDetector(
            num_node_features=sample_data.num_node_features,
            num_edge_features=sample_data.edge_attr.size(1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            temporal_memory = None
            prev_pred = None
            
            for t in range(self.time_windows):
                self.optimizer.zero_grad()
                data = self.temporal_graph.get_time_window_data(t)
                
                predictions, temporal_memory = self.model(data, temporal_memory)
                
                # Base loss
                base_loss = F.mse_loss(predictions, torch.zeros_like(predictions))
                
                # Add temporal consistency loss only if we have previous predictions
                loss = base_loss
                if prev_pred is not None:
                    # Ensure tensors have the same size since we're using fixed edges
                    temporal_consistency = F.mse_loss(predictions, prev_pred)
                    loss += 0.5 * temporal_consistency
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                prev_pred = predictions.detach()  # Store current predictions for next iteration
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Avg Loss: {total_loss/self.time_windows:.4f}')

    def detect_anomalies(self, threshold=0.8):
        self.model.eval()
        temporal_anomalies = {}
        temporal_memory = None
        
        with torch.no_grad():
            for t in range(self.time_windows):
                data = self.temporal_graph.get_time_window_data(t)
                scores, temporal_memory = self.model(data, temporal_memory)
                temporal_anomalies[t] = (scores.numpy(), scores.numpy() > threshold)
        
        return temporal_anomalies

    def visualize_temporal_anomalies(self, temporal_anomalies):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        time_points = [0, self.time_windows//3, 2*self.time_windows//3, self.time_windows-1]
        
        for idx, t in enumerate(time_points):
            G = nx.Graph()
            data = self.temporal_graph.get_time_window_data(t)
            edge_list = data.edge_index.t().numpy()
            scores, anomaly_mask = temporal_anomalies[t]
            
            for i, (u, v) in enumerate(edge_list):
                G.add_edge(u, v, anomalous=anomaly_mask[i])
            
            pos = nx.spring_layout(G)
            
            normal_edges = [(u, v) for (u, v, d) in G.edges(data=True) if not d['anomalous']]
            nx.draw_networkx_edges(G, pos, edgelist=normal_edges, alpha=0.2, ax=axes[idx])
            
            anomalous_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['anomalous']]
            nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, edge_color='r', width=2, ax=axes[idx])
            
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', ax=axes[idx])
            axes[idx].set_title(f"Time Window {t}")
        
        plt.suptitle("Temporal Communication Graph Anomalies")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    graph = TemporalCommunicationGraph(num_nodes=500, avg_degree=10, time_windows=24)
    graph.generate_nodes().generate_temporal_edges()
    
    detector = TemporalAnomalyDetectionFramework(graph)
    detector.train(epochs=50)
    
    temporal_anomalies = detector.detect_anomalies()
    detector.visualize_temporal_anomalies(temporal_anomalies)