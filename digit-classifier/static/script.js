const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const predictedDigitEl = document.getElementById('predictedDigit');
const confidenceChart = document.getElementById('confidenceChart');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

function clearCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultsSection.style.opacity = '0';
}

clearCanvas();

ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000';

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches?.[0].clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0].clientY) - rect.top;
    return [x, y];
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const [x, y] = getPos(e);
    [lastX, lastY] = [x, y];
    
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const [x, y] = getPos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);

function initChart() {
    confidenceChart.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'chart-row';
        row.innerHTML = `
            <div class="digit-label">${i}</div>
            <div class="bar-container">
                <div class="bar-fill" id="bar-${i}"></div>
            </div>
            <div class="conf-value" id="conf-${i}">0%</div>
        `;
        confidenceChart.appendChild(row);
    }
}

initChart();

predictBtn.addEventListener('click', async () => {
    const dataURL = canvas.toDataURL('image/png');
    
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataURL })
        });
        
        const data = await response.json();
        
        predictedDigitEl.textContent = data.prediction;
        
        for (let i = 0; i < 10; i++) {
            const conf = data.confidences[i];
            const pct = (conf * 100).toFixed(1);
            
            const bar = document.getElementById(`bar-${i}`);
            const text = document.getElementById(`conf-${i}`);
            
            bar.style.width = `${pct}%`;
            text.textContent = `${pct}%`;
            
            const intensity = 200 - Math.floor(conf * 200);
            bar.style.backgroundColor = `rgb(${intensity}, ${intensity}, ${intensity})`;
        }
        
        resultsSection.style.opacity = '1';
        
    } catch (error) {
        console.error("Prediction error:", error);
        alert("Failed to get prediction. Is the backend running?");
    } finally {
        predictBtn.textContent = 'Predict';
        predictBtn.disabled = false;
    }
});
