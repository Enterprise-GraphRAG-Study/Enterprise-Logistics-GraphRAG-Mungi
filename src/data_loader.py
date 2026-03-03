import torch
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_node_tensor(device):
    url = "https://huggingface.co/datasets/azminetoushikwasi/SupplyGraph/resolve/main/Raw%20Dataset/Homogenoeus/Nodes/Nodes.csv"
    df = pd.read_csv(url)
    
    # ID 매핑 및 텐서 변환
    node_ids = df['Node'].unique()
    id_map = {id_str: i for i, id_str in enumerate(node_ids)}
    numeric_indices = [id_map[id_str] for id_str in df['Node']]
    
    return torch.tensor(numeric_indices, dtype=torch.long).to(device)