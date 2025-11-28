# app.py
from flask import Flask, request, jsonify
from embedding import get_document_embedding # Lấy hàm đã sửa
from simhash import find_duplicates_simhash # Lấy hàm đã sửa
from minhash import find_duplicates_minhash # Lấy hàm đã sửa
from faiss_search import find_duplicates_faiss # Lấy hàm đã sửa
import numpy as np
import clustering # Giả định bạn đã có logic clustering trong file này

app = Flask(__name__)

# Dữ liệu mẫu (chỉ để demo)
SAMPLE_TEXTS = [
    "Việt Nam là một nước xã hội chủ nghĩa.",
    "Việt Nam là một nước xã hội chủ nghĩa với thủ đô là Hà Nội.",
    "Hệ thống định vị toàn cầu GPS rất quan trọng.",
    "Bóng đá là môn thể thao vua.",
    "GPS (Global Positioning System) là một hệ thống quan trọng."
]

@app.route('/deduplicate', methods=['POST'])
def deduplicate_texts():
    """
    Chạy pipeline phát hiện và loại bỏ văn bản trùng lặp.
    """
    try:
        # Lấy dữ liệu từ request (hoặc dùng dữ liệu mẫu)
        data = request.get_json()
        texts = data.get('texts', SAMPLE_TEXTS)
        method = data.get('method', 'faiss')
        
        if not texts:
            return jsonify({"error": "Không có văn bản nào được cung cấp."}), 400
        
        results = {}
        
        # 1. Trích xuất Embedding [cite: 7]
        embeddings = get_document_embedding(texts)
        
        # 2. Tìm kiếm trùng lặp theo phương pháp đã chọn
        if method == 'faiss':
            # Ngưỡng khoảng cách L2 (cần tuning)
            duplicates = find_duplicates_faiss(embeddings, distance_threshold=0.5) 
        elif method == 'simhash':
            # Ngưỡng khoảng cách Hamming (cần tuning)
            duplicates = find_duplicates_simhash(embeddings, threshold=5) 
        elif method == 'minhash':
            # MinHash hoạt động trên văn bản thô, không dùng embedding
            duplicates = find_duplicates_minhash(texts)
        else:
            return jsonify({"error": f"Phương pháp '{method}' không được hỗ trợ."}), 400

        # 3. Gom cụm và chọn văn bản đại diện [cite: 28]
        # Giả định clustering.run_clustering có thể xử lý kết quả từ các phương pháp băm/tìm kiếm
        cluster_info = clustering.run_clustering(texts, duplicates, method) 
        
        results = {
            "method_used": method,
            "total_documents": len(texts),
            "duplicates_found": len(duplicates),
            "cluster_summary": cluster_info,
            "raw_pairs": duplicates
        }
        
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

if __name__ == '__main__':
    # Chạy Flask app
    print("Khởi động server Flask...")
    # Cần đảm bảo mô hình embedding được load thành công
    app.run(debug=True)