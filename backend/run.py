"""
Backend chính - Flask API xử lý phát hiện trùng lặp
"""
import os
import time
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

# Import các module xử lý
from utils import extract_text_from_file
from embedding import get_embeddings_from_texts
from simhash import find_duplicates_simhash
from minhash import find_duplicates_minhash
from faiss_search import find_duplicates_faiss
from clustering import process_clustering
from export_word import create_deduplication_report

# ===== SETUP FLASK =====
app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_docs'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'doc', 'docx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global cache để lưu file tạo ra
generated_files = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===== API ENDPOINTS =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra backend có chạy không"""
    return jsonify({
        'status': 'ok',
        'message': 'Backend is running',
        'version': '1.0.0'
    })


@app.route('/api/process', methods=['POST'])
def process_file():
    """
    Endpoint chính: upload file → embedding → dedup → clustering → export docx
    """
    
    start_time = time.time()
    
    try:
        # ===== BƯỚC 1: KIỂM TRA FILE =====
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        method = request.form.get('method', 'all')
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'error': f'Định dạng file không được hỗ trợ. Chỉ hỗ trợ: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        print(f"\n{'='*60}")
        print(f"📥 Nhận file: {file.filename}")
        print(f"{'='*60}")
        
        # ===== BƯỚC 2: ĐỌC NỘI DUNG FILE =====
        try:
            texts = extract_text_from_file(file)
        except Exception as e:
            return jsonify({'error': f'Lỗi đọc file: {str(e)}'}), 400
        
        if len(texts) < 2:
            return jsonify({'error': f'File phải chứa ít nhất 2 văn bản (hiện có {len(texts)})'}), 400
        
        # ===== BƯỚC 3: TẠO EMBEDDINGS =====
        print(f"\n📊 Bước 1/4: Tạo Embeddings")
        print(f"{'='*60}")
        
        try:
            embeddings = get_embeddings_from_texts(texts, batch_size=32)
            embeddings = embeddings.astype(np.float32)
        except Exception as e:
            return jsonify({'error': f'Lỗi tạo embedding: {str(e)}'}), 500
        
        # ===== BƯỚC 4: PHÁT HIỆN TRÙNG LẶP =====
        print(f"\n🔍 Bước 2/4: Phát Hiện Trùng Lặp")
        print(f"{'='*60}")
        
        methods_to_run = {}
        
        if method == 'all':
            methods_to_run = {
                'simhash': 'SimHash',
                'minhash': 'MinHash',
                'faiss': 'FAISS'
            }
        else:
            method_names = {
                'simhash': 'SimHash',
                'minhash': 'MinHash',
                'faiss': 'FAISS'
            }
            methods_to_run = {method: method_names.get(method, method)}
        
        results = {}
        word_files = []
        
        for method_key, method_name in methods_to_run.items():
            try:
                # Phát hiện trùng lặp theo phương pháp
                if method_key == 'faiss':
                    pairs = find_duplicates_faiss(embeddings, top_k=5, similarity_threshold=0.85)
                
                elif method_key == 'simhash':
                    pairs = find_duplicates_simhash(embeddings, nbits=128, bands=8, hamming_threshold=15)
                
                elif method_key == 'minhash':
                    pairs = find_duplicates_minhash(texts, num_perm=128, jaccard_threshold=0.5)
                
                else:
                    raise ValueError(f"Phương pháp '{method_key}' không được hỗ trợ")
                
                # ===== BƯỚC 5: PHÂN CỤM =====
                print(f"\n🔗 Bước 3/4: Phân Cụm ({method_name})")
                print(f"{'='*60}")
                
                clustering_result = process_clustering(
                    pairs,
                    texts,
                    embeddings,
                    representative_method='centroid'
                )
                
                # ===== BƯỚC 6: XUẤT DOCX =====
                print(f"\n📄 Bước 4/4: Xuất Báo Cáo ({method_name})")
                print(f"{'='*60}")
                
                # Tạo tên file
                timestamp = int(time.time())
                doc_filename = f"report_{method_key}_{timestamp}.docx"
                doc_path = os.path.join(OUTPUT_FOLDER, doc_filename)
                
                # Chuẩn bị performance data
                elapsed_time = time.time() - start_time
                performance = {
                    'Phương pháp': method_name,
                    'Thời gian xử lý': f"{elapsed_time:.2f}s",
                    'Số văn bản': f"{len(texts)} văn bản",
                    'Số cặp tương tự': f"{len(pairs)} cặp"
                }
                
                # Xuất Word
                create_deduplication_report(
                    clustering_result,
                    method_name,
                    doc_path,
                    performance
                )
                
                word_files.append(doc_filename)
                generated_files[doc_filename] = doc_path
                
                # Chuẩn bị kết quả trả về
                results[method_name] = {
                    'success': True,
                    'stats': clustering_result['stats'],
                    'clusters': clustering_result['clusters'],
                    'performance': performance
                }
                
                print(f"✓ {method_name} xử lý thành công")
            
            except Exception as e:
                print(f"❌ Lỗi với {method_name}: {str(e)}")
                results[method_name] = {'error': str(e)}
        
        # ===== TRẢ VỀ KẾT QUẢ =====
        print(f"\n{'='*60}")
        print(f"✓ XỬ LÝ HOÀN THÀNH")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'methods': results,
            'word_files': word_files,
            'timestamp': timestamp
        })
    
    except Exception as e:
        print(f"\n❌ Lỗi backend: {str(e)}")
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download file báo cáo Word"""
    try:
        if filename not in generated_files:
            return jsonify({'error': 'File không tồn tại'}), 404
        
        file_path = generated_files[filename]
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File không tìm thấy'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    
    except Exception as e:
        return jsonify({'error': f'Lỗi download: {str(e)}'}), 500


# ===== ERROR HANDLERS =====

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File quá lớn (tối đa 50MB)'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Lỗi server nội bộ'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint không tìm thấy'}), 404


# ===== MAIN =====

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 KHỞI ĐỘNG SERVER PHÁT HIỆN TRÙNG LẶP VĂN BẢN")
    print("="*60)
    print(f"📍 Backend: http://localhost:5000")
    print(f"📍 API Base: http://localhost:5000/api")
    print(f"📍 Health Check: http://localhost:5000/api/health")
    print("="*60 + "\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Tắt reloader để tránh lỗi multi-processing
    )