const imageUpload = document.getElementById('imageUpload');
const resultDiv = document.getElementById('result');
const canvas = document.getElementById('canvas');

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights'),
  faceapi.nets.faceLandmark68Net.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights')
]).then(start);

function start() {
  imageUpload.addEventListener('change', async () => {
    const file = imageUpload.files[0];
    if (!file) return;
    const img = await loadImage(file);
    canvas.width = img.width;
    canvas.height = img.height;
    faceapi.matchDimensions(canvas, { width: img.width, height: img.height });

    const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks();

    resultDiv.textContent = `Faces detected: ${detections.length}`;
    faceapi.draw.drawDetections(canvas, faceapi.resizeResults(detections, { width: img.width, height: img.height }));
    faceapi.draw.drawFaceLandmarks(canvas, faceapi.resizeResults(detections, { width: img.width, height: img.height }));
  });
}

function loadImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = URL.createObjectURL(file);
  });
}
