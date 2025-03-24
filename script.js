const imageContainer = document.createElement('div');
imageContainer.style.position = 'relative';
document.body.append(imageContainer);

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start);

async function start() {
    document.body.append('Face API Models Loaded');
    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    // Fetch and process image from Flask API
    getImageFromServer(faceMatcher);
}

async function getImageFromServer(faceMatcher) {
    try {
        const response = await fetch("http://127.0.0.1:5000/get_image");
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        processImage(imageUrl, faceMatcher);
    } catch (error) {
        console.error("Error fetching image from server:", error);
    }
}

async function processImage(imageUrl, faceMatcher) {
    const image = await faceapi.fetchImage(imageUrl);
    
    // Remove previous images/canvases if they exist
    imageContainer.innerHTML = '';
    imageContainer.append(image);

    const canvas = faceapi.createCanvasFromMedia(image);
    imageContainer.append(canvas);

    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);

    // Detect faces in the image
    const detections = await faceapi.detectAllFaces(image)
                                    .withFaceLandmarks()
                                    .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map(d => ({
        label: faceMatcher.findBestMatch(d.descriptor).toString(),
        box: {
            x: d.detection.box.x,
            y: d.detection.box.y,
            width: d.detection.box.width,
            height: d.detection.box.height
        }
    }));

    // Draw detected faces and labels on the canvas
    results.forEach(result => {
        const drawBox = new faceapi.draw.DrawBox(result.box, { label: result.label });
        drawBox.draw(canvas);
    });

    // Send detected faces to Flask server
    sendDetectionsToServer(results);
}

async function sendDetectionsToServer(detectedFaces) {
    try {
        const response = await fetch("http://127.0.0.1:5000/receive_detections", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ faces: detectedFaces })
        });

        const responseData = await response.json();
        console.log("Response from Flask:", responseData);
    } catch (error) {
        console.error("Error sending detections to server:", error);
    }
}

function loadLabeledImages() {
    const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark'];
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/LAMENTIS1/face_recog/master/labeled_images/${label}/${i}.jpg`);
                const detections = await faceapi.detectSingleFace(img)
                                                .withFaceLandmarks()
                                                .withFaceDescriptor();
                descriptions.push(detections.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}
