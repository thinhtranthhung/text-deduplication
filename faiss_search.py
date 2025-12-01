"""
Module tìm kiếm trùng lặp sử dụng FAISS
"""
import numpy as np
import faiss
from typing import List, Tuple


def find_duplicates_faiss(
    embeddings: np.ndarray,
    top_k: int = 5,
    similarity_threshold: float = 0.85
) -> List[Tuple[int, int, float]]:
    """
    Tìm các cặp văn bản tương tự sử dụng FAISS
    
    Args:
        embeddings: numpy array shape (n_docs, embedding_dim) - float32
        top_k: Số láng giềng gần nhất để kiểm tra
        similarity_threshold: Ngưỡng cosine similarity
    
    Returns:
        List các tuple (doc_id_1, doc_id_2, similarity_score)
    """
    
    if embeddings is None or embeddings.size == 0:
        print("⚠️  Embeddings trống")
        return []
    
    n_docs, embedding_dim = embeddings.shape
    print(f"🔍 FAISS: Tìm kiếm trùng lặp trong {n_docs} văn bản (dim={embedding_dim})")
    
    # Copy embeddings để không thay đổi original
    embeddings_copy = embeddings.copy().astype(np.float32)
    
    # Normalize cho inner product = cosine similarity
    faiss.normalize_L2(embeddings_copy)
    
    # Tạo index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_copy)
    
    # Search
    distances, indices = index.search(embeddings_copy, min(top_k, n_docs))
    
    # Lấy kết quả
    similar_pairs = set()
    results = []
    
    for i in range(n_docs):
        for rank in range(1, min(top_k, len(indices[i]))):
            j = int(indices[i][rank])
            
            # Bỏ qua self-comparison hoặc kết quả không hợp lệ
            if j == -1 or i == j:
                continue
            
            sim_score = float(distances[i][rank])
            
            # Chỉ giữ cặp vượt ngưỡng
            if sim_score >= similarity_threshold:
                pair = tuple(sorted([i, j]))
                
                if pair not in similar_pairs:
                    similar_pairs.add(pair)
                    results.append((pair[0], pair[1], sim_score))
    
    # Sắp xếp theo similarity giảm dần
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"✓ Tìm được {len(results)} cặp tương tự (ngưỡng: {similarity_threshold})")
    return results


if __name__ == '__main__':
    # Test
    from embedding import get_embeddings_from_texts
    
    test_texts = [
        "Việt Nam là một nước xã hội chủ nghĩa",
        "Việt Nam là một nước xã hội chủ nghĩa với thủ đô Hà Nội",
        "Hà Nội là thủ đô của Việt Nam",
        "Python là ngôn ngữ lập trình phổ biến",
    ]
    
    embeddings = get_embeddings_from_texts(test_texts)
    results = find_duplicates_faiss(embeddings, similarity_threshold=0.7)
    
    print("\nKết quả:")
    for i, j, sim in results:
        print(f"  ({i}, {j}): {sim:.4f} - '{test_texts[i][:50]}...' <-> '{test_texts[j][:50]}...'")