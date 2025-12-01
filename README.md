````markdown
# Hệ thống Phát hiện và Loại bỏ Văn bản Trùng lặp (Deduplication System)

Hệ thống này cung cấp một API Flask mạnh mẽ để xử lý và loại bỏ các văn bản **trùng lặp** hoặc **gần giống** trong một tập dữ liệu lớn. Dự án hiện thực và so sánh ba phương pháp phát hiện tương đồng chính: **FAISS (dựa trên Embedding)**, **SimHash (dựa trên Embedding)**, và **MinHash (dựa trên Shingling)**.

Dữ liệu đầu vào (TXT, CSV, JSON, DOCX) sẽ được xử lý qua các bước: Trích xuất Text -> Tạo Embeddings -> Phát hiện Trùng lặp -> Phân cụm -> Xuất báo cáo Word.

---

## Mục tiêu và Tính năng Chính

Đây là một đề tài mở rộng trong lĩnh vực **Cấu trúc Dữ liệu và Giải thuật** và **Học sâu (Deep Learning)** nhằm:

1.  **Trích xuất Đặc trưng (Embedding):** Sử dụng các mô hình học sâu như `Sentence-Transformers` (mặc định là `all-MiniLM-L6-v2`) để ánh xạ văn bản thành vector đặc trưng cố định chiều.
2.  **Phát hiện Trùng lặp:** Triển khai 3 phương pháp chính:
    * **FAISS:** Tìm kiếm Tương đồng Gần đúng (Approximate Nearest Neighbor - ANN) trên không gian Embedding.
    * **SimHash:** Băm đặc trưng Embedding xuống không gian nhị phân và sử dụng LSH để tìm kiếm với khoảng cách Hamming.
    * **MinHash:** Sử dụng K-shingling và LSH để ước lượng độ tương đồng Jaccard cho văn bản thô.
3.  **Phân cụm & Chọn Đại diện:** Gom các văn bản tương đồng thành cụm bằng thuật toán Union-Find và chọn ra một văn bản đại diện (ngắn gọn nhất, nhiều từ khóa nhất, hoặc gần centroid nhất).
4.  **Báo cáo Tự động:** Tạo báo cáo chi tiết dưới dạng file `.docx` bao gồm thống kê, hiệu năng, và chi tiết từng cụm văn bản trùng lặp.
5.  **Hỗ trợ đa định dạng:** Xử lý các file đầu vào như `.txt`, `.csv`, `.json`, `.doc`, `.docx`.

---

## Cấu trúc Hệ thống

Hệ thống được xây dựng dưới dạng một API Flask, với các module Python độc lập cho từng bước xử lý:

| File | Mô tả |
| :--- | :--- |
| `run.py` | **Backend Chính (Flask API)**. Xử lý logic nghiệp vụ, quản lý luồng (`/api/process`, `/api/download`) và cấu hình server. |
| `embedding.py` | Module tạo **Embedding** vector cho văn bản bằng thư viện `sentence-transformers`. |
| `faiss_search.py` | Tìm kiếm trùng lặp sử dụng **FAISS** (dựa trên Cosine Similarity của Embeddings). |
| `simhash.py` | Tìm kiếm trùng lặp sử dụng **SimHash** (dựa trên Hamming Distance của băm Embedding). |
| `minhash.py` | Tìm kiếm trùng lặp sử dụng **MinHash + LSH** (dựa trên Jaccard Similarity của k-shingles). |
| `clustering.py` | Thực hiện **Phân cụm** (Union-Find) và logic **chọn Văn bản Đại diện** (`centroid`, `shortest`, `longest`). |
| `export_word.py` | Module tạo file **báo cáo DOCX** từ kết quả phân cụm. |
| `utils.py` | Hàm tiện ích để đọc và trích xuất text từ các định dạng file đầu vào khác nhau (`.txt`, `.csv`, `.json`, `.docx`). |

---

## Cài đặt và Chạy Thử

### 1. Yêu cầu Hệ thống

* Python 3.8+
* Hệ điều hành hỗ trợ PyTorch/Sentence-Transformers (khuyến nghị Linux/Windows/macOS).
* Nếu sử dụng **Google Colab** hoặc server có GPU, FAISS sẽ được tối ưu hóa.

### 2. Cài đặt Thư viện

Tạo môi trường ảo và cài đặt các dependencies cần thiết:

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# Cài đặt các thư viện
pip install -r requirements.txt
# (Giả định requirements.txt chứa: flask, flask-cors, numpy, sentence-transformers, torch, faiss-cpu, datasketch, python-docx, tqdm, werkzeug)
````

**Lưu ý:** Thư viện `faiss` được sử dụng là `faiss-cpu` hoặc `faiss-gpu` tùy thuộc vào hệ thống.

### 3\. Khởi động Server

Chạy file `run.py` để khởi động ứng dụng Flask API:

```bash
python run.py
```

Server sẽ chạy trên cổng **8000** (đã được sửa trong file `run.py`):

```
KHỞI ĐỘNG SERVER PHÁT HIỆN TRÙNG LẶP VĂN BẢN
============================================================
📍 Backend: http://localhost:8000
📍 API Base: http://localhost:8000/api
📍 Health Check: http://localhost:8000/api/health
============================================================
```

-----

## API Endpoint

### 1\. Health Check

  * **Endpoint:** `GET /api/health`
  * **Mô tả:** Kiểm tra trạng thái hoạt động của server.

### 2\. Xử lý File

Đây là endpoint chính để upload file và thực hiện toàn bộ pipeline.

  * **Endpoint:** `POST /api/process`
  * **Content-Type:** `multipart/form-data`
  * **Parameters:**
    | Tên | Loại | Mặc định | Mô tả |
    | :--- | :--- | :--- | :--- |
    | `file` | `File` | N/A | File đầu vào (TXT, CSV, JSON, DOC, DOCX). |
    | `method` | `String` | `all` | Phương pháp phát hiện trùng lặp: `faiss`, `simhash`, `minhash`, hoặc `all` (chạy tất cả). |
  * **Phản hồi thành công (200):**
    ```json
    {
      "success": true,
      "methods": {
        "FAISS": { /* kết quả thống kê của FAISS */ },
        "SimHash": { /* kết quả thống kê của SimHash */ },
        "MinHash": { /* kết quả thống kê của MinHash */ }
      },
      "word_files": ["report_faiss_1678886400.docx", /* ... */],
      "timestamp": 1678886400
    }
    ```

### 3\. Tải Báo Cáo

  * **Endpoint:** `GET /api/download/<filename>`
  * **Mô tả:** Tải file báo cáo `.docx` đã được tạo ra sau khi xử lý.

-----

## Luồng Xử Lý Chi Tiết (`/api/process`)

Pipeline xử lý được thực hiện theo 6 bước chính:

1.  **Kiểm tra File:** Xác nhận file được upload hợp lệ và định dạng được hỗ trợ.
2.  **Đọc Nội dung:** Trích xuất danh sách văn bản (`texts`) từ file đầu vào (`utils.py`).
3.  **Tạo Embeddings:** Sử dụng `SentenceTransformer` để tạo ma trận Embeddings (`embedding.py`).
4.  **Phát hiện Trùng lặp (`find_duplicates_*`):**
      * **FAISS:** `faiss_search.find_duplicates_faiss(embeddings)`
      * **SimHash:** `simhash.find_duplicates_simhash(embeddings)`
      * **MinHash:** `minhash.find_duplicates_minhash(texts)`
5.  **Phân cụm (`clustering.py`):** Dùng Union-Find trên các cặp trùng lặp được tìm thấy, sau đó chọn văn bản đại diện cho mỗi cụm.
6.  **Xuất DOCX (`export_word.py`):** Tạo báo cáo chi tiết, bao gồm thống kê hiệu năng và danh sách các cụm văn bản trùng lặp/đại diện.

-----

## Đánh giá và So sánh

Dự án này cho phép so sánh rõ ràng hiệu quả, tốc độ, và chi phí bộ nhớ giữa các phương pháp băm tự xây dựng và FAISS:

| Phương pháp | Loại đặc trưng | Metric Tương đồng | Tốc độ | Độ chính xác |
| :--- | :--- | :--- | :--- | :--- |
| **FAISS** | Vector Embedding (Float32) | Cosine Similarity (Inner Product) | Nhanh (với Index ANN) | Cao |
| **SimHash** | Băm Nhị phân (Bits) | Hamming Distance | Rất nhanh (với LSH) | Phụ thuộc vào số Bits |
| **MinHash** | Tập Shingle (Text Thô) | Jaccard Similarity | Nhanh (với LSH) | Phụ thuộc vào k-shingles |

Báo cáo cuối cùng (`.docx`) sẽ cung cấp các thống kê cần thiết (thời gian chạy, số lượng cụm, tỷ lệ loại bỏ) để người dùng có thể đánh giá và nhận xét ưu nhược điểm của từng phương pháp trong bối cảnh ứng dụng thực tế.
