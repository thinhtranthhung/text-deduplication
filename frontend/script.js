const API_BASE = 'http://localhost:5000/api';

let selectedFile = null;
let selectedMethod = 'all';

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');

// File handlers
dropZone.onclick = () => fileInput.click();

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
};

dropZone.ondragleave = () => {
    dropZone.classList.remove('dragover');
};

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
};

fileInput.onchange = (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
};

function handleFile(file) {
    const allowedTypes = ['.txt', '.csv', '.json', '.doc', '.docx'];
    const fileName = file.name.toLowerCase();
    const hasValidExt = allowedTypes.some(ext => fileName.endsWith(ext));
    
    if (!hasValidExt) {
        showError('Định dạng file không được hỗ trợ. Vui lòng upload: TXT, CSV, JSON, DOC, DOCX');
        return;
    }
    
    if (file.size > 50 * 1024 * 1024) {
        showError('File quá lớn. Tối đa 50MB');
        return;
    }
    
    selectedFile = file;
    document.getElementById('fileName').textContent = `📄 ${file.name}`;
    document.getElementById('fileSize').textContent = `Kích thước: ${(file.size / 1024).toFixed(2)} KB`;
    fileInfo.classList.add('show');
    processBtn.disabled = false;
    errorMessage.classList.remove('show');
}

// Method selection
document.querySelectorAll('.method-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedMethod = btn.dataset.method;
    };
});

// Process
processBtn.onclick = async () => {
    if (!selectedFile) return;
    
    processBtn.disabled = true;
    progressSection.classList.add('show');
    resultsSection.classList.remove('show');
    errorMessage.classList.remove('show');
    
    try {
        await processFile();
    } catch (error) {
        showError(error.message);
        progressSection.classList.remove('show');
        processBtn.disabled = false;
    }
};

async function processFile() {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('method', selectedMethod);
    
    updateProgress(10, 'Đang upload file...');
    
    const response = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Lỗi xử lý');
    }
    
    updateProgress(40, 'Đang tạo embeddings...');
    const data = await response.json();
    
    if (!data.success) {
        throw new Error('Xử lý không thành công');
    }
    
    updateProgress(70, 'Đang phát hiện trùng lặp...');
    await sleep(500);
    
    updateProgress(100, 'Hoàn thành!');
    await sleep(500);
    
    showResults(data);
}

function updateProgress(percent, text) {
    const fill = document.getElementById('progressFill');
    fill.style.width = percent + '%';
    fill.textContent = percent + '%';
    document.getElementById('progressText').textContent = text;
}

function showResults(data) {
    progressSection.classList.remove('show');
    resultsSection.classList.add('show');

    // Download links
    const downloadLinks = document.getElementById('downloadLinks');
    downloadLinks.innerHTML = '';

    if (data.word_files && data.word_files.length > 0) {
        data.word_files.forEach(file => {
            const button = document.createElement('button');
            button.className = 'download-btn';
            button.textContent = `📄 ${file.replace('report_', '').replace(`_${data.timestamp}`, '')}`;
            button.onclick = () => {
                window.location.href = `${API_BASE}/download/${file}`;
            };
            downloadLinks.appendChild(button);
        });
    }

    // Tabs & Stats
    const tabs = document.getElementById('methodTabs');
    const results = document.getElementById('methodResults');
    tabs.innerHTML = '';
    results.innerHTML = '';

    const methods = data.methods ? Object.keys(data.methods) : [];

    methods.forEach((method, i) => {
        const tab = document.createElement('button');
        tab.className = 'method-tab' + (i === 0 ? ' active' : '');
        tab.textContent = method;
        tab.onclick = () => switchMethod(method);
        tabs.appendChild(tab);

        const resultDiv = document.createElement('div');
        resultDiv.id = `result-${method}`;
        resultDiv.style.display = i === 0 ? 'block' : 'none';
        resultDiv.innerHTML = renderResult(data.methods[method]);
        results.appendChild(resultDiv);
    });
}

function switchMethod(method) {
    document.querySelectorAll('.method-tab').forEach(t => {
        t.classList.toggle('active', t.textContent === method);
    });
    document.querySelectorAll('[id^="result-"]').forEach(r => {
        r.style.display = r.id === `result-${method}` ? 'block' : 'none';
    });
}

function renderResult(methodData) {
    if (methodData.error) {
        return `<div style="color: red; padding: 20px;">❌ Lỗi: ${methodData.error}</div>`;
    }

    const stats = methodData.stats || {};

    let html = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats.total_docs || 0}</div>
                <div class="stat-label">Tổng văn bản</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.n_clusters || 0}</div>
                <div class="stat-label">Số cụm</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.n_removed || 0}</div>
                <div class="stat-label">Văn bản loại</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.n_kept || 0}</div>
                <div class="stat-label">Văn bản giữ lại</div>
            </div>
        </div>

        <div style="background: #f0f2ff; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
            <p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">
                📊 Tỷ lệ loại bỏ: ${stats.removal_rate ? (stats.removal_rate * 100).toFixed(1) : 0}%
            </p>
            <p style="color: #666; font-size: 14px;">
                Tải file Word để xem chi tiết các cụm trùng lặp và danh sách văn bản
            </p>
        </div>
    `;

    return html;
}

function showError(msg) {
    errorMessage.textContent = '❌ ' + msg;
    errorMessage.classList.add('show');
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// Reset
document.getElementById('resetBtn').onclick = () => {
    location.reload();
};

// Check backend on load
window.addEventListener('load', () => {
    fetch(`${API_BASE}/health`)
        .then(r => r.json())
        .then(d => console.log('✓ Backend:', d))
        .catch(err => {
            console.error('❌ Backend error:', err);
            showError('Không kết nối được Backend! Hãy chạy: python run.py');
        });
});