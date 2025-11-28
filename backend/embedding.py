"""
Module tạo embeddings từ text sử dụng SentenceTransformers
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class TextEmbedder:
    """Class để tạo embeddings cho văn bản"""
    
    _instance = None
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Khởi tạo model embedding (singleton pattern)
        
        Args:
            model_name: Tên model từ sentence-transformers
        """
        if TextEmbedder._instance is not None:
            self.model = TextEmbedder._instance.model
            self.model_name = TextEmbedder._instance.model_name
            return
        
        print(f"📦 Đang tải model {model_name}...")
        
        # Thiết lập device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        
        TextEmbedder._instance = self
        print(f"✓ Model {model_name} đã sẵn sàng")
    
    def embed_texts(self, texts, batch_size=64):
        """
        Tạo embeddings cho danh sách văn bản
        
        Args:
            texts: List các văn bản (string)
            batch_size: Số văn bản xử lý mỗi lần
        
        Returns:
            numpy array shape (n_texts, embedding_dim) - dtype float32
        """
        if not texts:
            raise ValueError("Danh sách văn bản rỗng")
        
        if len(texts) > 10000:
            print(f"⚠️  Cảnh báo: Embedding {len(texts)} văn bản có thể mất nhiều thời gian")
        
        print(f"📊 Tạo embeddings cho {len(texts)} văn bản (batch_size={batch_size})...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # Normalize sau trong FAISS
        )
        
        embeddings = embeddings.astype(np.float32)
        
        print(f"✓ Hoàn tất. Shape: {embeddings.shape}")
        return embeddings
    
    def get_embedding_dim(self):
        """Lấy chiều của embedding vector"""
        return self.model.get_sentence_embedding_dimension()


def get_embeddings_from_texts(texts: list, model_name='all-MiniLM-L6-v2', batch_size=64) -> np.ndarray:
    """
    Function tiện ích để tạo embeddings từ danh sách text
    
    Args:
        texts: List các văn bản string
        model_name: Tên model SentenceTransformer
        batch_size: Kích thước batch
    
    Returns:
        numpy array embeddings (float32)
    """
    embedder = TextEmbedder(model_name)
    return embedder.embed_texts(texts, batch_size)


if __name__ == '__main__':
    # Test
    test_texts = [
        "Việt Nam là một nước xã hội chủ nghĩa",
        "Việt Nam là một nước có thủ đô Hà Nội",
        "Python là ngôn ngữ lập trình phổ biến"
    ]
    
    embeddings = get_embeddings_from_texts(test_texts)
    print(f"\nTest embeddings shape: {embeddings.shape}")
    print(f"Type: {type(embeddings)}, dtype: {embeddings.dtype}")