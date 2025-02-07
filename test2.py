import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from py2neo import Graph, Node, Relationship


class AdvancedCommunicationGraph:
    def __init__(self, num_nodes=1000, avg_degree=8):
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.nodes = []
        self.edges = []
        self.clearance_levels = ['L1', 'L2', 'L3', 'L4']
        self.departments = ['ENG', 'HR', 'FIN', 'IT', 'SALES']

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

    def generate_edges(self):
        G = nx.barabasi_albert_graph(self.num_nodes, self.avg_degree//2)
        for u, v in G.edges():
            self.edges.append({
                'source': u,
                'target': v,
                'comms_volume': np.random.poisson(20),
                'encrypted_ratio': np.random.beta(2, 5),
                'sensitive_keywords': np.random.poisson(1.5),
                'time_diff': np.abs(np.random.normal(0, 3)),
                'clearance_diff': abs(self.clearance_levels.index(self.nodes[u]['clearance']) - 
                                    self.clearance_levels.index(self.nodes[v]['clearance']))
            })
        return self

    def to_pyg_data(self):
        # Process node features
        node_df = pd.DataFrame(self.nodes)
        categorical_features = ['department', 'clearance']
        numeric_features = ['risk_score', 'login_attempts', 'access_level', 'tenure', 'auth_failures']
        
        # One-hot encode categorical features
        node_df = pd.get_dummies(node_df, columns=categorical_features)
        node_features = node_df[numeric_features + 
                              [c for c in node_df.columns if c.startswith('department_') or c.startswith('clearance_')]]
        
        # Process edge features
        edge_df = pd.DataFrame(self.edges)
        edge_index = torch.tensor([edge_df['source'], edge_df['target']], dtype=torch.long)
        edge_features = edge_df.drop(columns=['source', 'target'])
        
        # Normalize features
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        edge_features = scaler.fit_transform(edge_features)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float)
        )


# GNN-Based Anomaly Detection Model
class GNNAnomalyDetector(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64):
        super().__init__()
        # Edge processing
        self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_dim)
        
        # Graph Attention Network layers
        self.conv1 = GATConv(num_node_features, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        
        # Anomaly scoring
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        # Process edge attributes
        edge_emb = self.edge_encoder(data.edge_attr)
        
        # Node embeddings
        x = F.relu(self.conv1(data.x, data.edge_index, edge_emb))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index, edge_emb)
        
        # Edge anomaly scores
        row, col = data.edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        return torch.sigmoid(self.scorer(edge_features)).squeeze()

# Training and Detection Pipeline
class AnomalyDetectionFramework:
    def __init__(self, data):
        self.data = data
        self.model = GNNAnomalyDetector(
            num_node_features=data.num_node_features,
            num_edge_features=data.edge_attr.size(1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(self.data)
            
            # Self-supervised loss (assume majority are normal)
            loss = F.mse_loss(predictions, torch.zeros_like(predictions))
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
                
    def detect_anomalies(self, threshold=0.8):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(self.data).numpy()
        return scores, scores > threshold
    
    def node2vec_embeddings(self, dimensions=64):
        """Generate node embeddings using Node2Vec"""
        G = nx.Graph()
        G.add_edges_from(self.data.edge_index.t().numpy())
        
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200)
        model = node2vec.fit(window=10, min_count=1)
        return {str(node): model.wv[str(node)] for node in G.nodes()}
    
    def visualize_anomalies(self, anomaly_mask):
        """Visualize graph with anomalous edges highlighted"""
        G = nx.Graph()
        edge_list = self.data.edge_index.t().numpy()
        
        for i, (u, v) in enumerate(edge_list):
            G.add_edge(u, v, anomalous=anomaly_mask[i])
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G)
        
        # Draw normal edges
        normal_edges = [(u, v) for (u, v, d) in G.edges(data=True) if not d['anomalous']]
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, alpha=0.2)
        
        # Draw anomalous edges
        anomalous_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['anomalous']]
        nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, edge_color='r', width=2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue')
        plt.title("Communication Graph with Anomalous Edges Highlighted")
        plt.show()


class Neo4jExporter(AdvancedCommunicationGraph):
    def __init__(self, num_nodes=1000, avg_degree=5):
        super().__init__(num_nodes, avg_degree)
        self.generate_nodes()
        self.generate_edges()
        
    def export_to_csv(self, nodes_file="nodes.csv", edges_file="relationships.csv"):
        """Export graph data to Neo4j-compatible CSV files"""
        # Prepare node data
        node_df = pd.DataFrame(self.nodes)
        
        # Prepare edge data
        edge_df = pd.DataFrame(self.edges)
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
            
# Example Usage
# if __name__ == "__main__":
#     # Generate synthetic graph
#     graph = AdvancedCommunicationGraph(num_nodes=500, avg_degree=10)
#     graph.generate_nodes().generate_edges()
#     pyg_data = graph.to_pyg_data()
    
#     # Initialize framework
#     detector = AnomalyDetectionFramework(pyg_data)
    
#     # Train GNN model
#     print("Training GNN anomaly detector...")
#     detector.train(epochs=50)
    
#     # Detect anomalies
#     scores, anomalies = detector.detect_anomalies(threshold=0.75)
#     print(f"Detected {sum(anomalies)} anomalous edges")
    
#     # Generate Node2Vec embeddings
#     embeddings = detector.node2vec_embeddings()
    
#     # Visualize results
#     detector.visualize_anomalies(anomalies)


if __name__ == "__main__":
    graph = AdvancedCommunicationGraph(num_nodes=500, avg_degree=10)
    graph.generate_nodes().generate_edges()
    pyg_data = graph.to_pyg_data()
    
    # Continue with detector initialization and training
    detector = AnomalyDetectionFramework(pyg_data)
    detector.train(epochs=50)
    scores, anomalies = detector.detect_anomalies()
    detector.visualize_anomalies(anomalies)

    # Export to Neo4j
    exporter = Neo4jExporter(num_nodes=500, avg_degree=8)
    nodes_df, edges_df = exporter.export_to_csv()

