const video = document.getElementById('video');
const loadingOverlay = document.getElementById('loading'); // Loading overlay element

// Load models asynchronously
Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  // Hide loading overlay once models are loaded
  loadingOverlay.style.display = 'none'; 

  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)

  // Set up webcam stream
  const videoStream = await navigator.mediaDevices.getUserMedia({ video: {} })
  video.srcObject = videoStream

  // When the video metadata is loaded, start the face detection
  video.onloadedmetadata = () => {
    video.play()
    detectFaces()
  }

  // Detect faces in the video stream
  async function detectFaces() {
    // Ensure the canvas has the same size as the video
    const canvas = faceapi.createCanvasFromMedia(video)
    document.body.append(canvas)
    const displaySize = { width: video.videoWidth, height: video.videoHeight }
    canvas.width = displaySize.width
    canvas.height = displaySize.height
    faceapi.matchDimensions(canvas, displaySize)

    // Set interval to detect faces and match them in real-time
    setInterval(async () => {
      const detections = await faceapi.detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors()

      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

      // Clear the canvas before drawing the updated results
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, canvas.width, canvas.height);  // Clears the entire canvas

      // Draw updated face recognition results
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box
        const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
        drawBox.draw(canvas)
      })
    }, 100) // Detect every 100ms (10fps)
  }
}

// Load labeled face images and create descriptors for face recognition
function loadLabeledImages() {
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark','vatsav','ashwini','vijay_ps','aswin_m']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/LAMENTIS1/face_recog/master/labeled_images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
