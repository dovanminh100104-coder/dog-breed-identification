document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    const loadingContainer = document.querySelector('.loading-container');
    
    const breedName = document.getElementById('breed-name');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const topList = document.getElementById('top-list');
    const historyBody = document.getElementById('history-body');

    // Load initial history
    fetchHistory();

    let selectedFile = null;

    // Trigger file input on click
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and Drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một file hình ảnh!');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultSection.classList.add('hidden');
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        resultSection.classList.remove('hidden');
        loadingContainer.classList.remove('hidden');
        resultContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
                fetchHistory(); // Refresh history
            } else {
                alert('Lỗi: ' + (data.error || 'Không thể nhận diện ảnh'));
                resultSection.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Có lỗi xảy ra khi kết nối với server.');
            resultSection.classList.add('hidden');
        } finally {
            loadingContainer.classList.add('hidden');
        }
    });

    function renderResults(data) {
        resultContent.classList.remove('hidden');
        breedName.textContent = data.prediction;
        
        // Update bar and text
        const confidence = data.confidence.toFixed(1);
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = confidence + '%';
        }, 100);
        confidenceText.textContent = confidence + '% Độ tự tin';

        // Update top 3 list
        topList.innerHTML = '';
        data.top_3.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${item.breed}</span>
                <span class="conf">${item.confidence.toFixed(1)}%</span>
            `;
            topList.appendChild(li);
        });

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    async function fetchHistory() {
        if (!historyBody) return;

        try {
            const response = await fetch('/history');
            const history = await response.json();
            
            historyBody.innerHTML = '';
            
            if (history.length === 0) {
                historyBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 30px;">Chưa có lịch sử nhận diện.</td></tr>';
                return;
            }

            history.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="time">${item.timestamp}</td>
                    <td class="breed-cell">${item.breed}</td>
                    <td class="confidence-cell">${item.confidence}</td>
                `;
                historyBody.appendChild(tr);
            });
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    }
});
