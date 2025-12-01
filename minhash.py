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

if __name__ == "__main__":
    print("="*70)
    print("TEST MINHASH – PHÁT HIỆN TRÙNG LẶP VĂN BẢN (DỰA TRÊN TỪ NGỮ)")
    print("="*70)

    # Dữ liệu test đa dạng: trùng 100%, gần giống, khác hoàn toàn
    test_texts = [
        "Python là ngôn ngữ lập trình phổ biến nhất hiện nay.",
        "Python là một ngôn ngữ lập trình rất phổ biến hiện nay.",
        "Python là ngôn ngữ lập trình phổ biến nhất hiện nay.",  # trùng 100% với dòng 0
        "Tôi yêu lập trình Python vì nó dễ học và rất mạnh mẽ.",
        "Tôi rất thích lập trình bằng Python vì nó dễ học và mạnh mẽ.",
        "Máy học và trí tuệ nhân tạo đang thay đổi thế giới.",
        "Trí tuệ nhân tạo cùng học máy đang thay đổi thế giới.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Thủ đô của Việt Nam chính là Hà Nội.",
        "Bóng đá là môn thể thao vua được yêu thích nhất trên thế giới.",
        "Bóng đá được coi là môn thể thao vua và được yêu thích nhất.",
        "Con mèo đang nằm ngủ trên ghế sofa.",
        "Con chó sủa rất to khi thấy người lạ.",
    ]

    print(f"Số văn bản test: {len(test_texts)}\n")

    # Test với nhiều ngưỡng Jaccard khác nhau
    thresholds = [0.5]

    for th in thresholds:
        print(f"\n{'='*25} NGƯỠNG JACCARD = {th} {'='*25}")
        duplicates = find_duplicates_minhash(
            texts=test_texts,
            num_perm=128,
            jaccard_threshold=th,
            k_shingles=5
        )
        
        if not duplicates:
            print("Không tìm thấy cặp nào trùng lặp ở ngưỡng này.")
            continue
            
        print(f"→ Tìm thấy {len(duplicates)} cặp (hiển thị tối đa 10 cặp đầu):\n")
        for rank, (i, j, sim) in enumerate(duplicates[:10], 1):
            print(f"{rank:2d}. [Jaccard ≈ {sim:.3f}] ID {i:2d} ↔ ID {j:2d}")
            print(f"    → \"{test_texts[i]}\"")
            print(f"    → \"{test_texts[j]}\"")
            print()

    print("\nTEST MINHASH HOÀN TẤT!")