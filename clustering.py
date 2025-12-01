"""
Module phân cụm và chọn văn bản đại diện
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class UnionFind:
    """Cấu trúc Union-Find để phân cụm"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def cluster_documents(
    pairs: List[Tuple[int, int, float]],
    n_docs: int
) -> Dict[int, List[int]]:
    """
    Phân cụm dựa trên các cặp tương tự
    
    Args:
        pairs: List các cặp (doc_id1, doc_id2, similarity)
        n_docs: Tổng số văn bản
    
    Returns:
        Dict {cluster_root: [doc_ids]}
    """
    uf = UnionFind(n_docs)
    
    for i, j, _ in pairs:
        uf.union(i, j)
    
    clusters = defaultdict(list)
    for doc_id in range(n_docs):
        root = uf.find(doc_id)
        clusters[root].append(doc_id)
    
    # Chỉ giữ clusters có > 1 văn bản
    return {k: v for k, v in clusters.items() if len(v) > 1}


def select_representative(
    cluster: List[int],
    texts: List[str],
    embeddings: np.ndarray = None,
    method: str = 'shortest'
) -> int:
    """
    Chọn văn bản đại diện cho cụm
    
    Args:
        cluster: List doc_ids trong cụm
        texts: Danh sách tất cả văn bản
        embeddings: Embeddings (optional, cho method='centroid')
        method: 'shortest', 'longest', 'centroid'
    
    Returns:
        Doc ID của văn bản đại diện
    """
    
    if len(cluster) == 1:
        return cluster[0]
    
    if method == 'centroid' and embeddings is not None:
        # Chọn văn bản gần centroid nhất
        cluster_vecs = embeddings[cluster]
        centroid = np.mean(cluster_vecs, axis=0)
        distances = np.linalg.norm(cluster_vecs - centroid, axis=1)
        return cluster[np.argmin(distances)]
    
    elif method == 'shortest':
        # Chọn văn bản ngắn nhất
        return min(cluster, key=lambda i: len(texts[i]))
    
    elif method == 'longest':
        # Chọn văn bản dài nhất
        return max(cluster, key=lambda i: len(texts[i]))
    
    else:
        return cluster[0]


def process_clustering(
    pairs: List[Tuple[int, int, float]],
    texts: List[str],
    embeddings: np.ndarray = None,
    representative_method: str = 'centroid'
) -> Dict:
    """
    Phân cụm và chọn đại diện cho mỗi cụm
    
    Returns:
        {
            'clusters': {
                cluster_id: {
                    'docs': [doc_ids],
                    'representative': int,
                    'documents': [
                        {'id': int, 'text': str, 'is_representative': bool}
                    ]
                }
            },
            'stats': {...}
        }
    """
    
    n_docs = len(texts)
    print(f"\nPhân cụm: {n_docs} văn bản, {len(pairs)} cặp tương tự")
    
    # Phân cụm
    clusters_raw = cluster_documents(pairs, n_docs)
    print(f" Tìm được {len(clusters_raw)} cụm trùng lặp")
    
    # Xử lý từng cụm
    clusters_output = {}
    all_duplicates = set()
    
    for cluster_id, doc_ids in clusters_raw.items():
        # Chọn đại diện
        representative_id = select_representative(
            doc_ids, texts, embeddings, representative_method
        )
        
        # Tạo danh sách documents cho cluster
        documents = []
        for doc_id in doc_ids:
            documents.append({
                'id': doc_id,
                'text': texts[doc_id],
                'is_representative': (doc_id == representative_id)
            })
            
            if doc_id != representative_id:
                all_duplicates.add(doc_id)
        
        clusters_output[cluster_id] = {
            'docs': doc_ids,
            'representative': representative_id,
            'size': len(doc_ids),
            'documents': documents
        }
    
    # Tính thống kê
    n_removed = len(all_duplicates)
    n_kept = n_docs - n_removed
    
    stats = {
        'total_docs': n_docs,
        'n_clusters': len(clusters_output),
        'n_removed': n_removed,
        'n_kept': n_kept,
        'n_pairs': len(pairs),
        'removal_rate': n_removed / n_docs if n_docs > 0 else 0
    }
    
    print(f" Thống kê:")
    print(f"   - Tổng văn bản: {stats['total_docs']}")
    print(f"   - Số cụm: {stats['n_clusters']}")
    print(f"   - Văn bản bị loại: {stats['n_removed']}")
    print(f"   - Văn bản giữ lại: {stats['n_kept']}")
    print(f"   - Tỷ lệ loại: {stats['removal_rate']:.1%}")
    
    return {
        'clusters': clusters_output,  # ← Đây phải là dict
        'stats': stats,
        'duplicates': sorted(list(all_duplicates)),
        'kept': [i for i in range(n_docs) if i not in all_duplicates]
    }
