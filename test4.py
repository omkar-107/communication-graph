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

class AdvancedCommunicationGraph:
    def __init__(self, num_nodes=1000, avg_degree=5):
        self.fake = Faker()
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.node_features = []
        self.edge_features = []
        self.edge_index = []
        self.clearance_levels = {
            'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4
        }
        
    def generate_nodes(self):
        for _ in range(self.num_nodes):
            node = {
                'email': self.fake.unique.email(),
                'department': random.choice(['ENG', 'HR', 'FIN', 'IT', 'SALES']),
                'clearance': random.choice(list(self.clearance_levels.keys())),
                'risk_score': np.random.beta(1, 5),
                'failed_logins': np.random.poisson(0.3),
                'access_level': random.randint(1, 5),
                'tenure': np.random.lognormal(2, 0.5)
            }
            self.node_features.append(node)
        return self
    
    def generate_edges(self):
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
            
            self.edge_features.append(edge)
            self.edge_index.append([u, v])
            
            # Add the same edge features for the reverse direction
            self.edge_features.append(edge)
            self.edge_index.append([v, u])
            
        return self
    
    def to_pyg_data(self):
        node_df = pd.DataFrame(self.node_features)
        edge_df = pd.DataFrame(self.edge_features)
        
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

class GraphAnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.edge_scores = None
        
    def train_gae(self, epochs=100):
        class Encoder(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, out_channels)
                self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
                self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = torch.nn.functional.elu(x)
                x = self.dropout(x)
                
                x = self.conv2(x, edge_index)
                x = self.bn2(x)
                x = torch.nn.functional.elu(x)
                x = self.dropout(x)
                
                return self.conv3(x, edge_index)
            
        class EdgeDecoder(torch.nn.Module):
            def __init__(self, in_channels, edge_attr_dim, dropout=0.2):
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_channels * 2, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(128, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(64, edge_attr_dim)
                )
                
            def forward(self, z, edge_index):
                row, col = edge_index
                edge_features = torch.cat([z[row], z[col]], dim=1)
                return self.mlp(edge_features)
                
        in_channels = self.data.num_node_features
        hidden_channels = 64
        out_channels = 32
        edge_attr_dim = self.data.edge_attr.size(1)
        
        encoder = Encoder(in_channels, hidden_channels, out_channels)
        decoder = EdgeDecoder(out_channels, edge_attr_dim)
        self.model = torch.nn.ModuleList([encoder, decoder])
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_loss = float('inf')
        patience = 15
        counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = self.model[0](self.data.x, self.data.edge_index)
            edge_attr_pred = self.model[1](z, self.data.edge_index)
            loss = torch.nn.functional.mse_loss(edge_attr_pred, self.data.edge_attr)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                print("Early stopping!")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        with torch.no_grad():
            z = self.model[0](self.data.x, self.data.edge_index)
            edge_attr_pred = self.model[1](z, self.data.edge_index)
            self.edge_scores = (edge_attr_pred - self.data.edge_attr).pow(2).mean(dim=1).numpy()
            
        return self
    
    def detect_structural_anomalies(self):
        detector = ECOD()
        detector.fit(self.data.x.numpy())
        return detector.decision_scores_
    
    def detect_edge_anomalies(self):
        edge_features = np.column_stack([
            self.edge_scores,
            self.data.edge_attr.numpy()
        ])
        
        detector = IForest(contamination=0.05)
        detector.fit(edge_features)
        return detector.decision_scores_
    
    def visualize_anomalies(self, top_k=20):
        G = nx.Graph()
        edge_scores = self.detect_edge_anomalies()
        edges = self.data.edge_index.t().numpy()
        # Only consider one direction of each edge
        unique_edges = set()
        filtered_edges = []
        filtered_scores = []
        
        for i, (u, v) in enumerate(edges):
            edge = tuple(sorted([u, v]))
            if edge not in unique_edges:
                unique_edges.add(edge)
                filtered_edges.append((u, v))
                filtered_scores.append(edge_scores[i])
        
        top_edges = sorted(zip(filtered_edges, filtered_scores),
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

class Neo4jExporter(AdvancedCommunicationGraph):
    def __init__(self, num_nodes=1000, avg_degree=5):
        super().__init__(num_nodes, avg_degree)
        self.generate_nodes()
        self.generate_edges()
        
    def export_to_csv(self, nodes_file="nodes.csv", edges_file="relationships.csv"):
        """Export graph data to Neo4j-compatible CSV files"""
        # Prepare node data
        node_df = pd.DataFrame(self.node_features)
        
        # Prepare edge data
        edge_df = pd.DataFrame(self.edge_features)
        edge_df = edge_df.rename(columns={
            'source': 'src',
            'target': 'dst',
            'comms_volume': 'comms',
            'encrypted_ratio': 'encrypted',
            'time_diff': 'time_diff_mean'
        })
        
        # Save to CSV
        node_df.to_csv(nodes_file, index=False)
        edge_df.to_csv(edges_file, index=False)
        
        # Print Cypher commands for importing
        print("\nNeo4j Import Commands:")
        print("\n1. First, load nodes:")
        print("""
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (:User {
    nodeId: toInteger(row.id),
    department: row.department,
    clearance: row.clearance,
    riskScore: toFloat(row.risk_score),
    loginAttempts: toInteger(row.login_attempts),
    accessLevel: toInteger(row.access_level),
    tenure: toFloat(row.tenure),
    authFailures: toInteger(row.auth_failures)
})
        """)
        
        print("\n2. Then, load relationships:")
        print("""
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (source:User {nodeId: toInteger(row.src)})
MATCH (target:User {nodeId: toInteger(row.dst)})
CREATE (source)-[:COMMUNICATES {
    comms: toInteger(row.comms),
    encrypted: toFloat(row.encrypted),
    sensitiveKeywords: toInteger(row.sensitive_keywords),
    timeDiffMean: toFloat(row.time_diff_mean),
    clearanceDiff: toInteger(row.clearance_diff)
}]->(target)
        """)
        
        return node_df, edge_df

    def export_to_neo4j(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """Export directly to Neo4j using py2neo"""
        try:
            # Connect to Neo4j
            graph = Graph(uri, auth=(user, password))
            
            # Clear existing data
            graph.run("MATCH (n) DETACH DELETE n")
            
            # Create nodes
            print("Creating nodes...")
            nodes = {}
            for node_data in self.nodes:
                node = Node("User",
                          nodeId=node_data['id'],
                          department=node_data['department'],
                          clearance=node_data['clearance'],
                          riskScore=float(node_data['risk_score']),
                          loginAttempts=int(node_data['login_attempts']),
                          accessLevel=int(node_data['access_level']),
                          tenure=float(node_data['tenure']),
                          authFailures=int(node_data['auth_failures']))
                graph.create(node)
                nodes[node_data['id']] = node
            
            # Create relationships
            print("Creating relationships...")
            for edge in self.edges:
                src_node = nodes[edge['source']]
                dst_node = nodes[edge['target']]
                rel = Relationship(src_node, "COMMUNICATES", dst_node,
                                 comms=int(edge['comms_volume']),
                                 encrypted=float(edge['encrypted_ratio']),
                                 sensitiveKeywords=int(edge['sensitive_keywords']),
                                 timeDiffMean=float(edge['time_diff']),
                                 clearanceDiff=int(edge['clearance_diff']))
                graph.create(rel)
                
            print("Export completed successfully!")
            
        except Exception as e:
            print(f"Error during export: {str(e)}")

if __name__ == "__main__":
    graph = AdvancedCommunicationGraph(num_nodes=500, avg_degree=8)
    graph.generate_nodes().generate_edges()
    pyg_data = graph.to_pyg_data()
    
    detector = GraphAnomalyDetector(pyg_data)
    detector.train_gae(epochs=1000)
    
    node_anomalies = detector.detect_structural_anomalies()
    edge_anomalies = detector.detect_edge_anomalies()
    
    print(f"Node anomaly scores range: {node_anomalies.min():.2f} - {node_anomalies.max():.2f}")
    print(f"Edge anomaly scores range: {edge_anomalies.min():.2f} - {edge_anomalies.max():.2f}")
    
    detector.visualize_anomalies()

    # Export to Neo4j
    exporter = Neo4jExporter(num_nodes=500, avg_degree=8)
    nodes_df, edges_df = exporter.export_to_csv()