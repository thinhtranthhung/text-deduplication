"""
Module tìm kiếm trùng lặp sử dụng SimHash
"""
import numpy as np
from typing import List, Tuple
import itertools
from collections import defaultdict


class SimHasher:
    """SimHash dựa trên embedding vectors"""
    
    def __init__(self, dim: int, nbits: int = 128, seed: int = 42):
        """
        Khởi tạo SimHasher
        
        Args:
            dim: Chiều embedding vector
            nbits: Số bit của SimHash (mặc định 128)
            seed: Random seed
        """
        self.dim = dim
        self.nbits = nbits
        self.seed = seed
        
        np.random.seed(seed)
        
        # Tạo random hyperplanes cho mỗi bit
        self.planes = np.random.randn(nbits, dim).astype(np.float32)
        self.planes = self.planes / (np.linalg.norm(self.planes, axis=1, keepdims=True) + 1e-8)
        
        print(f"✓ SimHash: {nbits} bits, dim={dim}")
    
    def hash(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Tính SimHash cho mảng embeddings
        
        Args:
            embeddings: shape (n, dim) - float32
        
        Returns:
            shape (n, nbits) - binary array
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        n, d = embeddings.shape
        if d != self.dim:
            raise ValueError(f"Chiều mismatch: {d} != {self.dim}")
        
        # Dot product với hyperplanes
        dots = np.dot(embeddings.astype(np.float32), self.planes.T)  # (n, nbits)
        
        # Convert to bits
        bits = (dots > 0).astype(np.uint8)  # (n, nbits)
        
        return bits
    
    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        """Tính Hamming distance giữa 2 hash vectors"""
        return np.count_nonzero(hash1 != hash2)


def find_duplicates_simhash(
    embeddings: np.ndarray,
    nbits: int = 128,
    bands: int = 8,
    hamming_threshold: int = 15
) -> List[Tuple[int, int, int]]:
    """
    Tìm các cặp văn bản tương tự sử dụng SimHash
    Args:
        embeddings: numpy array shape (n_docs, embedding_dim) - float32
        nbits: Số bit của SimHash
        bands: Số band cho LSH (mặc định 8)
        hamming_threshold: Ngưỡng Hamming distance
    Returns:
        List các tuple (doc_id_1, doc_id_2, hamming_distance)
    """
    
    n_docs, embedding_dim = embeddings.shape
    print(f"SimHash: Xử lý {n_docs} văn bản (nbits={nbits}, bands={bands})")

    # Tạo hasher
    hasher = SimHasher(dim=embedding_dim, nbits=nbits, seed=42)
    
    # Hash toàn bộ embeddings
    print("Bước 1: Hash embeddings...")
    hashes = hasher.hash(embeddings.astype(np.float32))  # (n_docs, nbits)
    
    # LSH với bands
    print("Bước 2: LSH indexing...")
    band_width = nbits // bands
    hash_tables = [defaultdict(list) for _ in range(bands)]
    
    for doc_id in range(n_docs):
        hash_vec = hashes[doc_id]
        for band_idx in range(bands):
            # Lấy bits của band này
            band_bits = hash_vec[band_idx * band_width:(band_idx + 1) * band_width]
            # Convert to integer
            band_hash = int(''.join(map(str, band_bits)), 2)
            hash_tables[band_idx][band_hash].append(doc_id)
    
    # Lấy candidate pairs từ LSH
    print("Bước 3: Finding candidates...")
    candidate_pairs = set()
    for band_table in hash_tables:
        for bucket in band_table.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidate_pairs.add(tuple(sorted(pair)))
    
    # Xác nhận từng cặp
    print(f"Bước 4: Verifying {len(candidate_pairs)} candidates...")
    results = []
    
    for i, j in candidate_pairs:
        hamming_dist = SimHasher.hamming_distance(hashes[i], hashes[j])
        
        if hamming_dist <= hamming_threshold:
            results.append((i, j, hamming_dist))
    
    # Sắp xếp theo Hamming distance tăng dần
    results.sort(key=lambda x: x[2])
    
    print(f"Tìm được {len(results)} cặp tương tự (ngưỡng Hamming: {hamming_threshold})")
    return results
