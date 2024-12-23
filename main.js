/*****************************************************
 * Global Variables
 *****************************************************/
// let model = null; // (Commented out) Will hold the loaded TF.js model

let accelData = [];          // Buffer for the accelerometer data
const SAMPLE_RATE = 50;      // Approx. sampling rate in Hz
const BUFFER_LENGTH_SEC = 2; // We want 2 seconds of data
const MAX_SAMPLES = SAMPLE_RATE * BUFFER_LENGTH_SEC;

/*****************************************************
 * Request Motion Permission (iOS >= 13)
 *****************************************************/
async function requestMotionPermission() {
  // Check if iOS 13+ style permission is needed
  if (typeof DeviceMotionEvent !== 'undefined' && 
      typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const response = await DeviceMotionEvent.requestPermission();
      if (response === 'granted') {
        console.log('DeviceMotion permission granted.');
        return true;
      } else {
        console.warn('DeviceMotion permission denied.');
        return false;
      }
    } catch (err) {
      console.error('Error requesting DeviceMotion permission:', err);
      return false;
    }
  } else {
    // Non-iOS or iOS < 13
    return true;
  }
}

/*****************************************************
 * Handle DeviceMotion Events
 *****************************************************/
function handleMotionEvent(event) {
  // For demonstration, we'll use the accelerationIncludingGravity.
  // If you want raw acceleration, use 'event.acceleration'.
  const { x, y, z } = event.accelerationIncludingGravity;

  // Store a single reading
  accelData.push({ x, y, z, t: performance.now() });

  // Keep only the last 2 seconds of data
  if (accelData.length > MAX_SAMPLES) {
    accelData.shift();
  }

  // For now, just display the latest reading in real-time
  document.getElementById('status').textContent =
    `Latest Reading: x=${x.toFixed(3)}, y=${y.toFixed(3)}, z=${z.toFixed(3)}`;
}

/*****************************************************
 * Start Collecting Accelerometer Data
 *****************************************************/
function startCollectingData() {
  window.addEventListener('devicemotion', handleMotionEvent, true);
}

/*****************************************************
 * (Commented Out) Load the TF.js Model
 *****************************************************/
// async function loadModel() {
//   try {
//     // model = await tf.loadGraphModel('model_web/model.json');
//     // console.log('Model loaded successfully.');
//   } catch (error) {
//     console.error('Error loading model:', error);
//   }
// }

/*****************************************************
 * (Commented Out) Run Inference
 *****************************************************/
// async function runInference() {
//   if (!model) {
//     console.warn('Model not loaded yet.');
//     return;
//   }
//   // ...
// }

/*****************************************************
 * Init / Main Function
 *****************************************************/
async function initDemo() {
  // 1. Request motion permission
  const granted = await requestMotionPermission();
  if (!granted) {
    document.getElementById('status').textContent = 'Permission for motion denied.';
    return;
  }

  // 2. Start collecting data
  startCollectingData();

  // (Commented Out) Model loading & inference
  // await loadModel();
  // setInterval(() => {
  //   runInference();
  // }, 2000);
}

// Attach initDemo to the start button click
document.getElementById('startButton').addEventListener('click', initDemo);
