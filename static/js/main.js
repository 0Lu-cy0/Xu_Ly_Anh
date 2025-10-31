// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const selectFileBtn = document.getElementById('selectFileBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);

// Button click - chỉ button này mở file dialog
selectFileBtn.addEventListener('click', (e) => {
  e.stopPropagation(); // Ngăn event bubble lên uploadBox
  fileInput.click();
});

// Drag & Drop
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('drag-over');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

// Click to upload (chỉ khi click vào vùng trống, không phải button)
uploadBox.addEventListener('click', (e) => {
  // Nếu click vào button thì bỏ qua (button tự xử lý)
  if (e.target === selectFileBtn || selectFileBtn.contains(e.target)) {
    return;
  }
  fileInput.click();
});

// Handle file selection
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleFile(file);
  }
}

// Handle file upload
function handleFile(file) {
  // Validate file type - Hỗ trợ nhiều định dạng
  const validTypes = [
    'image/jpeg', 'image/jpg', 'image/png', 'image/bmp',
    'image/tiff', 'image/webp', 'image/gif', 'image/x-icon',
    'image/vnd.microsoft.icon', 'image/svg+xml'
  ];

  // Kiểm tra extension nếu MIME type không khớp
  const fileName = file.name.toLowerCase();
  const validExtensions = ['jpg', 'jpeg', 'jpe', 'jfif', 'png', 'bmp', 'dib',
    'tiff', 'tif', 'webp', 'gif', 'ico',
    'ppm', 'pgm', 'pbm', 'pnm', 'svg', 'svgz'];

  const hasValidType = validTypes.includes(file.type);
  const hasValidExt = validExtensions.some(ext => fileName.endsWith('.' + ext));

  if (!hasValidType && !hasValidExt) {
    alert('Vui lòng chọn file ảnh hợp lệ!\nHỗ trợ: JPG, PNG, BMP, TIFF, WebP, GIF, ICO, PPM, PGM, PBM, SVG');
    return;
  }

  // Validate file size (16MB)
  if (file.size > 16 * 1024 * 1024) {
    alert('File quá lớn! Vui lòng chọn file nhỏ hơn 16MB');
    return;
  }

  // Show loading
  uploadBox.style.display = 'none';
  results.style.display = 'none';
  loading.style.display = 'block';

  // Upload file
  uploadFile(file);
}

// Upload file to server
function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
    .then(response => {
      console.log('Response status:', response.status);
      return response.json();
    })
    .then(data => {
      console.log('Response data:', data);
      if (data.success) {
        displayResults(data);
      } else {
        alert('Lỗi: ' + (data.error || 'Không xác định'));
        resetUpload();
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('Đã xảy ra lỗi khi upload file!');
      resetUpload();
    });
}

// Display results
function displayResults(data) {
  const result = data.result;
  const vis = data.visualizations;

  // Hide loading, show results
  loading.style.display = 'none';
  results.style.display = 'block';

  // Status badge
  const statusBadge = document.getElementById('statusBadge');
  const statusIcon = document.getElementById('statusIcon');
  const statusText = document.getElementById('statusText');

  if (result.label === 'TƯƠI') {
    statusBadge.className = 'status-badge fresh';
    statusIcon.textContent = '🍎';
    statusText.textContent = 'TƯƠI';
  } else if (result.label === 'TRUNG BÌNH') {
    statusBadge.className = 'status-badge medium';
    statusIcon.textContent = '🍊';
    statusText.textContent = 'TRUNG BÌNH';
  } else {
    statusBadge.className = 'status-badge rotten';
    statusIcon.textContent = '🍂';
    statusText.textContent = 'HỎNG';
  }

  // Health score
  const healthScore = document.getElementById('healthScore');
  const scoreFill = document.getElementById('scoreFill');

  healthScore.textContent = result.health_score + '/100';
  scoreFill.style.width = result.health_score + '%';

  // Confidence
  document.getElementById('confidence').textContent = result.confidence;

  // Reason
  const reasonDiv = document.getElementById('reason');
  reasonDiv.innerHTML = '<strong>Lý do:</strong> ' + result.reason;

  // Metrics
  document.getElementById('colorScore').textContent = result.metrics.color_health.toFixed(1) + '/100';
  document.getElementById('textureScore').textContent = result.metrics.texture_smoothness.toFixed(1) + '/100';
  document.getElementById('smoothScore').textContent = result.metrics.surface_quality.toFixed(1) + '/100';
  document.getElementById('defectCount').textContent = result.metrics.defect_count + ' vết';

  // Images
  document.getElementById('imgOriginal').src = vis.original;
  document.getElementById('imgHue').src = vis.hue;
  document.getElementById('imgSaturation').src = vis.saturation;
  document.getElementById('imgValue').src = vis.value;
  document.getElementById('imgEdges').src = vis.edges;
  document.getElementById('imgDefects').src = vis.defects;

  // Scroll to results
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Reset upload
function resetUpload() {
  uploadBox.style.display = 'block';
  loading.style.display = 'none';
  results.style.display = 'none';
  fileInput.value = '';

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());
