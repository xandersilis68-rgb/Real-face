const imageUpload = document.getElementById('imageUpload');
const detectAllBtn = document.getElementById('detectAllBtn');
const imagesContainer = document.getElementById('imagesContainer');

let poseModel, handposeModel;
let imageBlocks = [];

// Body skeleton connections (MoveNet order)
const BODY_CONNECTIONS = [
  [0,1],[1,3],[0,2],[2,4],
  [0,5],[5,7],[7,9],[0,6],[6,8],[8,10],
  [5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
];

// Facial landmark connections
const FACE_CONNECTIONS = {
  jaw: [0,16],
  leftEye: [36,41],
  rightEye: [42,47],
  leftBrow: [17,21],
  rightBrow: [22,26],
  nose: [27,35],
  outerMouth: [48,59],
  innerMouth: [60,67]
};

// Hand landmark connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20]
];

// Load models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights'),
  faceapi.nets.faceLandmark68Net.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights'),
  (async () => {
    poseModel = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
  })(),
  (async () => {
    handposeModel = await handpose.load();
  })()
]).then(() => {
  detectAllBtn.disabled = false;
});

imageUpload.addEventListener('change', async () => {
  imageBlocks = [];
  imagesContainer.innerHTML = '';
  const files = Array.from(imageUpload.files);
  if (!files.length) {
    detectAllBtn.disabled = true;
    return;
  }
  detectAllBtn.disabled = false;

  for (let idx = 0; idx < files.length; idx++) {
    const file = files[idx];
    const block = document.createElement('div');
    block.className = 'images-block';
    block.id = `imgBlock${idx}`;

    // Image element
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.style.display = 'block';
    img.onload = () => {
      // Set up canvas after image loads
      canvas.width = img.width;
      canvas.height = img.height;
    };

    // Canvas element
    const canvasWrapper = document.createElement('div');
    canvasWrapper.className = 'canvas-wrapper';
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    canvas.style.width = '100%';
    canvasWrapper.appendChild(img);
    canvasWrapper.appendChild(canvas);

    // Result div
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result';

    // Detect button
    const detectBtn = document.createElement('button');
    detectBtn.textContent = 'Detect Landmarks';
    detectBtn.onclick = async () => {
      await detectLandmarks(img, canvas, resultDiv);
    };

    // Download button
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = 'Download Result';
    downloadBtn.disabled = true;
    downloadBtn.onclick = () => {
      const link = document.createElement('a');
      link.download = 'landmark_result.png';
      link.href = canvas.toDataURL();
      link.click();
    };

    block.appendChild(canvasWrapper);
    block.appendChild(detectBtn);
    block.appendChild(downloadBtn);
    block.appendChild(resultDiv);

    imagesContainer.appendChild(block);

    imageBlocks.push({
      img, canvas, resultDiv, detectBtn, downloadBtn
    });
  }
});

// Detect Landmarks for all images
detectAllBtn.addEventListener('click', async () => {
  for (const block of imageBlocks) {
    await detectLandmarks(block.img, block.canvas, block.resultDiv, block.downloadBtn);
  }
});

async function detectLandmarks(img, canvas, resultDiv, downloadBtn) {
  // Make sure image loaded
  if (!img.complete || img.naturalWidth === 0) {
    resultDiv.textContent = "Image not loaded!";
    return;
  }
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Face landmarks
  const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  let numFaces = detections.length;
  resultDiv.innerHTML = `Faces detected: ${numFaces}<br/>`;
  detections.forEach((det, idx) => {
    // Bounding box
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    const box = det.detection.box;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Facial landmarks
    ctx.fillStyle = "blue";
    det.landmarks.positions.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Connections
    Object.values(FACE_CONNECTIONS).forEach(([start, end]) => {
      ctx.strokeStyle = "blue";
      ctx.beginPath();
      for (let i = start; i < end; i++) {
        const p1 = det.landmarks.positions[i];
        const p2 = det.landmarks.positions[i+1];
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
      }
      ctx.stroke();
    });

    // Confidence
    resultDiv.innerHTML += `Face ${idx+1} confidence: ${(det.detection.score*100).toFixed(1)}%<br/>`;
  });

  // Body pose
  const poses = await poseModel.estimatePoses(img);
  poses.forEach((pose, idx) => {
    resultDiv.innerHTML += `Body ${idx+1} confidence: ${(pose.score*100).toFixed(1)}%<br/>`;
    pose.keypoints.forEach(kp => {
      if (kp.score > 0.3) {
        ctx.fillStyle = "green";
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;
    BODY_CONNECTIONS.forEach(([i, j]) => {
      const kp1 = pose.keypoints[i];
      const kp2 = pose.keypoints[j];
      if (kp1.score > 0.3 && kp2.score > 0.3) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }
    });
  });

  // Hand landmarks
  const predictions = await handposeModel.estimateHands(img, false);
  resultDiv.innerHTML += `Hands detected: ${predictions.length}<br/>`;
  predictions.forEach((hand, idx) => {
    hand.landmarks.forEach(([x, y], i) => {
      ctx.fillStyle = "orange";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.strokeStyle = "orange";
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([start, end]) => {
      const p1 = hand.landmarks[start];
      const p2 = hand.landmarks[end];
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
    });
    resultDiv.innerHTML += `Hand ${idx+1} confidence: ${(hand.handInViewConfidence*100).toFixed(1)}%<br/>`;
  });

  if (downloadBtn) downloadBtn.disabled = false;
}
