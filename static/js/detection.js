document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const startBtn = document.getElementById('startBtn');
    const predictionText = document.getElementById('prediction');
    const historyBox = document.getElementById('history');
    
    let isDetecting = false;
    let stream = null;
    let detectionInterval = null;
    
    // Start or stop detection
    startBtn.addEventListener('click', function() {
        if (isDetecting) {
            stopDetection();
        } else {
            startDetection();
        }
    });
    
    // Start webcam and detection
    async function startDetection() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile if available
                } 
            });
            
            video.srcObject = stream;
            
            // Wait for video to be ready
            video.onloadedmetadata = function() {
                // Set canvas size to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Start detection loop
                isDetecting = true;
                startBtn.textContent = 'Stop Detection';
                startBtn.classList.remove('btn-primary');
                startBtn.classList.add('btn-secondary');
                
                // Run detection every 1 second
                detectionInterval = setInterval(detectSkinCancer, 1000);
            };
        } catch (err) {
            console.error('Error accessing webcam:', err);
            alert('Could not access webcam. Please ensure you have granted camera permissions.');
        }
    }
    
    // Stop detection and release webcam
    function stopDetection() {
        isDetecting = false;
        startBtn.textContent = 'Start Detection';
        startBtn.classList.remove('btn-secondary');
        startBtn.classList.add('btn-primary');
        
        clearInterval(detectionInterval);
        predictionText.textContent = 'Prediction: None';
        
        // Stop all tracks from the stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        video.srcObject = null;
    }
    
    // Capture frame and send to server for prediction
    function detectSkinCancer() {
        if (!isDetecting) return;
        
        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data as base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server for prediction
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Prediction error:', data.error);
                return;
            }
            
            // Update prediction text
            const resultText = `${data.class} (${data.confidence.toFixed(2)}%)`;
            predictionText.textContent = `Prediction: ${resultText}`;
            
            // Add to history
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.textContent = `${data.username} - ${resultText}`;
            historyBox.prepend(historyItem);
            
            // Limit history items
            if (historyBox.children.length > 20) {
                historyBox.removeChild(historyBox.lastChild);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});