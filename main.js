/*****************************************************
 * Global Variables
 *****************************************************/
let model = null;            // Will hold the loaded TF.js model
let accelData = [];          // Will store the raw accelerometer data
const TARGET_SAMPLES = 52;   // 52 samples total
const SAMPLING_RATE = 26;    // 26 Hz
const COLLECTION_TIME = 2000; // 2 seconds in milliseconds

// We'll store the interval/timeout so we can stop sampling after 2s
let samplingInterval = null;

/*****************************************************
 * Request Motion Permission (for iOS >= 13)
 *****************************************************/
async function requestMotionPermission() {
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
    // Non-iOS or older iOS
    return true;
  }
}

/*****************************************************
 * Load the TF.js Model (converted from .h5)
 *****************************************************/
async function loadModel() {
  try {
    // Adjust the path if needed:
    // e.g., 'model_web/model.json'
    model = await tf.loadLayersModel('model_web/model.json');
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

/*****************************************************
 * Start Accelerometer Collection
 *****************************************************/
function startCollectingData() {
  // Reset old data
  accelData = [];

  // Listen to devicemotion events
  window.addEventListener('devicemotion', handleMotionEvent, true);

  // Stop collecting after 2 seconds
  setTimeout(() => {
    stopCollectingData();
  }, COLLECTION_TIME);
}

/*****************************************************
 * Stop Accelerometer Collection
 *****************************************************/
function stopCollectingData() {
  window.removeEventListener('devicemotion', handleMotionEvent, true);

  // If we don't have 52 samples yet, it means the sampling rate was too low or the device was slow
  // We might handle that with zero-padding or repeated samples, etc.
  if (accelData.length < TARGET_SAMPLES) {
    // Example: zero-pad until we have 52
    while (accelData.length < TARGET_SAMPLES) {
      accelData.push({ x: 0, y: 0, z: 0 });
    }
  } else if (accelData.length > TARGET_SAMPLES) {
    // If we have more than 52, let's just take the last 52
    // Or you could do a more sophisticated resample. For now, keep it simple.
    accelData = accelData.slice(-TARGET_SAMPLES);
  }

  document.getElementById('status').textContent = 
    `Collected ${accelData.length} samples. Running inference...`;

  runInference();
}

/*****************************************************
 * Handle DeviceMotion
 *****************************************************/
function handleMotionEvent(event) {
  const { x, y, z } = event.accelerationIncludingGravity;
  
  // Store a single reading
  accelData.push({ x, y, z });

  // Optional: For debugging, show how many samples we have so far
  document.getElementById('status').textContent = 
    `Collecting samples... (${accelData.length})`;
}

/*****************************************************
 * Run Inference
 *****************************************************/
async function runInference() {
  if (!model) {
    console.warn('Model not loaded yet.');
    document.getElementById('status').textContent = 'Model not loaded.';
    return;
  }

  // Build a data array of shape [52, 3, 1]
  // (Each sample: [x, y, z], then we add an extra dimension for "channels" = 1)
  const dataArray = accelData.map(d => [[d.x], [d.y], [d.z]]); 
  // Now dataArray is shape [52, 3, 1] in a JavaScript sense

  // Convert to Tensor: shape [52, 3, 1]
  let inputTensor = tf.tensor3d(dataArray);

  // Reshape to [1, 52, 3, 1] because your model expects "None, 52, 3, 1"
  // "None" = batch dimension
  inputTensor = inputTensor.reshape([1, 52, 3, 1]);

  // Make prediction
  const outputTensor = model.predict(inputTensor);

  // Convert the result to JavaScript array
  const predictions = await outputTensor.data();

  // Cleanup
  outputTensor.dispose();
  inputTensor.dispose();

  console.log('Predictions:', predictions);

  // Display the predictions
  // If your model returns more than one output, you might iterate over them
  document.getElementById('status').textContent =
    'Inference result: ' + JSON.stringify(Array.from(predictions));
}

/*****************************************************
 * Main Init Function
 *****************************************************/
async function initDemo() {
  // 1. Request motion permission (for iOS)
  const granted = await requestMotionPermission();
  if (!granted) {
    document.getElementById('status').textContent = 
      'Permission for motion denied.';
    return;
  }

  // 2. Load the model
  await loadModel();

  // 3. Start collecting data
  //    We'll collect 2 seconds of data for each click, then automatically stop and run inference.
  startCollectingData();
}

// Attach init to the "Start" button
document.getElementById('startButton').addEventListener('click', initDemo);
