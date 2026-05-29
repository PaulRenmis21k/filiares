/* =============================================
   Potato Disease Classifier - Main JavaScript
   ============================================= */

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const selectBtn = document.getElementById('select-btn');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const previewFilename = document.getElementById('preview-filename');
    const previewFilesize = document.getElementById('preview-filesize');
    const changeBtn = document.getElementById('change-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultClass = document.getElementById('result-class');
    const resultConfidence = document.getElementById('result-confidence');
    const resultBadge = document.getElementById('result-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const predictionBars = document.getElementById('prediction-bars');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const dismissError = document.getElementById('dismiss-error');

    let selectedFile = null;

    // Classification colors
    const classColors = {
        'Early Blight': '#e67e22',
        'Late Blight': '#e74c3c',
        'Healthy': '#27ae60'
    };

    const classIcons = {
        'Early Blight': '🍂',
        'Late Blight': '🍁',
        'Healthy': '🌿'
    };

    // ---- Event Handlers ----

    // Click upload zone to select file
    uploadZone.addEventListener('click', (e) => {
        if (e.target !== selectBtn && !selectBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Change image
    changeBtn.addEventListener('click', () => {
        resetPreview();
        fileInput.click();
    });

    // Classify
    classifyBtn.addEventListener('click', () => {
        if (selectedFile) {
            classifyImage(selectedFile);
        }
    });

    // Dismiss error
    dismissError.addEventListener('click', () => {
        errorMessage.style.display = 'none';
    });

    // ---- Feedback ----
    const feedbackSection = document.getElementById('feedback-section');
    const feedbackOptions = document.getElementById('feedback-options');
    const feedbackBtns = document.querySelectorAll('.feedback-btn');
    const confirmFeedbackBtn = document.getElementById('confirm-feedback-btn');
    const feedbackSuccess = document.getElementById('feedback-success');

    let lastPredictionId = null;
    let selectedFeedbackClass = null;

    feedbackBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            feedbackBtns.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedFeedbackClass = btn.dataset.classId;
        });
    });

    confirmFeedbackBtn.addEventListener('click', async () => {
        if (!lastPredictionId || !selectedFeedbackClass) return;

        confirmFeedbackBtn.disabled = true;
        confirmFeedbackBtn.innerHTML = '<div class="spinner-sm"></div> Enviando...';

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_id: lastPredictionId,
                    confirmed_class: selectedFeedbackClass
                })
            });

            if (response.ok) {
                feedbackSection.querySelector('.feedback-title').style.display = 'none';
                feedbackSection.querySelector('.feedback-subtitle').style.display = 'none';
                feedbackOptions.style.display = 'none';
                confirmFeedbackBtn.style.display = 'none';
                feedbackSuccess.style.display = 'block';
            } else {
                const err = await response.json();
                showError(err.detail || 'Error al enviar feedback');
            }
        } catch (error) {
            showError('Error al enviar feedback: ' + error.message);
        } finally {
            confirmFeedbackBtn.disabled = false;
        }
    });

    // ---- Functions ----

    function handleFileSelect(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tiff|webp)$/i)) {
            showError('Formato de archivo no válido. Usa JPG, PNG o BMP.');
            return;
        }

        // Validate file size (10 MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            showError('El archivo es demasiado grande. Máximo 10 MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = formatFileSize(file.size);
            uploadZone.style.display = 'none';
            previewSection.style.display = 'block';
            resultsContainer.style.display = 'none';
            loadingSpinner.style.display = 'none';
            errorMessage.style.display = 'none';
            classifyBtn.style.display = 'inline-flex';
        };
        reader.readAsDataURL(file);
    }

    function resetPreview() {
        selectedFile = null;
        previewSection.style.display = 'none';
        resultsContainer.style.display = 'none';
        loadingSpinner.style.display = 'none';
        errorMessage.style.display = 'none';
        uploadZone.style.display = 'block';
        fileInput.value = '';
    }

    async function classifyImage(file) {
        // Show loading
        loadingSpinner.style.display = 'block';
        resultsContainer.style.display = 'none';
        classifyBtn.style.display = 'none';
        errorMessage.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Error ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            showError(error.message || 'Error al clasificar la imagen. Intenta de nuevo.');
            loadingSpinner.style.display = 'none';
            classifyBtn.style.display = 'inline-flex';
        }
    }

    function displayResults(data) {
        loadingSpinner.style.display = 'none';
        resultsContainer.style.display = 'block';

        const top = data.predictions[0];
        const color = classColors[top.class_name] || '#6c5ce7';
        const icon = classIcons[top.class_name] || '🍃';

        // Main result with animation
        resultBadge.style.background = `${color}20`;
        resultBadge.innerHTML = `<span class="result-icon">${icon}</span>`;
        resultBadge.style.animation = 'none';
        void resultBadge.offsetHeight; // Trigger reflow
        resultBadge.style.animation = 'pulseIn 0.5s ease';

        resultClass.style.color = color;
        resultClass.textContent = top.class_name;

        const confidencePercent = (top.confidence * 100).toFixed(1);
        resultConfidence.textContent = `Confianza: ${confidencePercent}%`;

        // Animated confidence bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;
        }, 100);

        // Prediction bars for all classes
        predictionBars.innerHTML = '';
        data.predictions.forEach((pred, index) => {
            const barColor = pred.color || classColors[pred.class_name] || '#6c5ce7';
            const pct = (pred.confidence * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = 'prediction-bar-item';
            item.style.animationDelay = `${index * 0.1}s`;

            item.innerHTML = `
                <span class="prediction-bar-label">${pred.class_name}</span>
                <div class="prediction-bar-track">
                    <div class="prediction-bar-fill" style="width: 0%; background: ${barColor};">
                        ${pct > 20 ? `${pct}%` : ''}
                    </div>
                </div>
                <span class="prediction-bar-value">${pct}%</span>
            `;

            predictionBars.appendChild(item);

            // Animate each bar
            setTimeout(() => {
                const fill = item.querySelector('.prediction-bar-fill');
                fill.style.width = `${pct}%`;
            }, 200 + index * 100);
        });

        // Setup feedback section
        lastPredictionId = data.prediction_id;
        selectedFeedbackClass = top.class_id;

        // Reset feedback UI
        feedbackSection.style.display = 'block';
        feedbackSection.querySelector('.feedback-title').style.display = 'block';
        feedbackSection.querySelector('.feedback-subtitle').style.display = 'block';
        feedbackOptions.style.display = 'flex';
        confirmFeedbackBtn.style.display = 'inline-flex';
        confirmFeedbackBtn.disabled = false;
        confirmFeedbackBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
                <path d="M16.7 5.3L7.7 14.3L3.7 10.3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Confirmar clasificación
        `;
        feedbackSuccess.style.display = 'none';

        // Pre-select the predicted class
        feedbackBtns.forEach(b => {
            b.classList.remove('selected');
            if (b.dataset.classId === top.class_id) {
                b.classList.add('selected');
            }
        });
    }

    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'flex';
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
});
