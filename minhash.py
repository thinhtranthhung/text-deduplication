"""
Module tìm kiếm trùng lặp sử dụng MinHash + LSH
"""
import re
from typing import List, Tuple
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH


def create_shingles(text: str, k: int = 5) -> set:
    """
    Tạo k-shingles từ text
    Args:
        text: Văn bản đầu vào
        k: Kích thước shingle
    Returns:
        Set các shingle
    """
    # Normalize text
    text = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text) < k:
        return {text}
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])
    
    return shingles

def find_duplicates_minhash(
    texts: List[str],
    num_perm: int = 128,
    jaccard_threshold: float = 0.5,
    k_shingles: int = 5
) -> List[Tuple[int, int, float]]:
    """
    Tìm các cặp văn bản tương tự sử dụng MinHash + LSH
    Args:
        texts: List các văn bản string
        num_perm: Số lượng permutation trong MinHash
        jaccard_threshold: Ngưỡng Jaccard similarity
        k_shingles: Kích thước k-shingles
    Returns:
        List các tuple (doc_id_1, doc_id_2, jaccard_similarity)
    """
    
    if not texts or len(texts) < 2:
        print("Danh sách văn bản không hợp lệ")
        return []
    
    n_docs = len(texts)
    print(f"MinHash: Xử lý {n_docs} văn bản (num_perm={num_perm}, k={k_shingles})")
    
    # Bước 1: Tạo MinHash cho mỗi văn bản
    print("Bước 1: Tạo MinHash signatures...")
    minhashes = []
    
    for idx, text in enumerate(tqdm(texts, desc="   MinHash")):
        shingles = create_shingles(text, k=k_shingles)
        
        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        
        minhashes.append(m)
    
    # Bước 2: LSH để tìm candidates
    print("Bước 2: LSH to find candidate pairs...")
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    
    for idx, m in enumerate(minhashes):
        lsh.insert(f"doc_{idx}", m)
    
    # Bước 3: Tìm candidate pairs
    candidate_pairs = set()
    for idx, m in enumerate(minhashes):
        candidates = lsh.query(m)
        for candidate_id in candidates:
            candidate_idx = int(candidate_id.split('_')[1])
            if idx < candidate_idx:
                candidate_pairs.add((idx, candidate_idx))
    
    # Bước 4: Kiểm tra chi tiết từng cặp
    print(f"Bước 3: Xác nhận {len(candidate_pairs)} candidate pairs...")
    
    results = []
    for (i, j) in tqdm(candidate_pairs, desc="   Verify"):
        jaccard_sim = minhashes[i].jaccard(minhashes[j])
        
        if jaccard_sim >= jaccard_threshold:
            results.append((i, j, jaccard_sim))
    
    # Sắp xếp theo similarity giảm dần
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Tìm được {len(results)} cặp tương tự (ngưỡng: {jaccard_threshold})")
    return results

