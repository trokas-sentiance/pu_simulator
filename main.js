/*****************************************************
 * Global Variables
 *****************************************************/
let model = null;            // Holds the loaded TF.js model
let accelData = [];          // Stores the raw accelerometer data
const TARGET_SAMPLES = 52;   // 52 samples total
const COLLECTION_TIME = 2000; // 2 seconds in milliseconds

// Flag to control continuous collection (if you ever want to stop)
let continueLoop = true;

/*****************************************************
 * Request Motion Permission (for iOS >= 13)
 *****************************************************/
async function requestMotionPermission() {
  if (
    typeof DeviceMotionEvent !== 'undefined' &&
    typeof DeviceMotionEvent.requestPermission === 'function'
  ) {
    try {
      const response = await DeviceMotionEvent.requestPermission();
      if (response === 'granted') {
        console.log('DeviceMotion permission granted (iOS).');
        return true;
      } else {
        console.warn('DeviceMotion permission denied (iOS).');
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
 * Load the TF.js Model (converted from .h5)
 *****************************************************/
async function loadModel() {
  try {
    // Make sure this path is correct for where your model.json is located
    model = await tf.loadLayersModel('model_web/model.json');
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

/*****************************************************
 * startLoop()
 * - Called once after permission + model load
 * - Begins the continuous 2-second data collection cycles
 *****************************************************/
function startLoop() {
  console.log('Starting continuous loop...');
  // Make sure we haven't turned off the loop
  if (!continueLoop) return;

  startCollectingData(); 
}

/*****************************************************
 * Start Accelerometer Collection (for 2 seconds)
 *****************************************************/
function startCollectingData() {
  console.log('startCollectingData(): Resetting accelData and starting listener...');
  accelData = [];

  window.addEventListener('devicemotion', handleMotionEvent, true);

  // After 2 seconds, stop collecting and run inference
  setTimeout(() => {
    stopCollectingData();
  }, COLLECTION_TIME);
}

/*****************************************************
 * Stop Accelerometer Collection
 *****************************************************/
function stopCollectingData() {
  console.log('stopCollectingData(): Removing devicemotion listener.');
  window.removeEventListener('devicemotion', handleMotionEvent, true);

  const numSamples = accelData.length;
  console.log(`Collected ${numSamples} samples in 2s.`);

  // Adjust sample count to exactly TARGET_SAMPLES
  if (numSamples < TARGET_SAMPLES) {
    // Zero-pad
    while (accelData.length < TARGET_SAMPLES) {
      accelData.push({ x: 0, y: 0, z: 0 });
    }
    console.log(`Padded from ${numSamples} to ${accelData.length} samples.`);
  } else if (numSamples > TARGET_SAMPLES) {
    // Truncate
    accelData = accelData.slice(-TARGET_SAMPLES);
    console.log(`Truncated from ${numSamples} to ${accelData.length} samples.`);
  }

  document.getElementById('status').textContent = 
    `Collected ${accelData.length} samples. Running inference...`;

  // Run inference asynchronously
  runInference().then(() => {
    // Once inference is done, immediately start the next 2-second cycle
    if (continueLoop) {
      startCollectingData();
    }
  });
}

/*****************************************************
 * Handle DeviceMotion
 *****************************************************/
function handleMotionEvent(event) {
  const { x, y, z } = event.accelerationIncludingGravity;

  // Push each reading
  accelData.push({ x, y, z });

  // Show how many samples so far
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

  // Build data array of shape [52, 3, 1]
  const dataArray = accelData.map(d => [[d.x], [d.y], [d.z]]);

  // Create tensor of shape [52, 3, 1]
  let inputTensor = tf.tensor3d(dataArray);

  // Reshape to [1, 52, 3, 1]
  inputTensor = inputTensor.reshape([1, TARGET_SAMPLES, 3, 1]);

  // Predict
  const outputTensor = model.predict(inputTensor);
  const predictions = await outputTensor.data();

  // Cleanup
  outputTensor.dispose();
  inputTensor.dispose();

  console.log('Predictions:', predictions);
  document.getElementById('status').textContent =
    'Inference result: ' + JSON.stringify(Array.from(predictions));
}

/*****************************************************
 * initDemo()
 * - Called once, e.g. at page load or from a button click
 *****************************************************/
async function initDemo() {
  // 1. iOS motion permission
  const granted = await requestMotionPermission();
  if (!granted) {
    document.getElementById('status').textContent = 
      'Permission for motion denied.';
    console.warn('Cannot proceed without motion permission.');
    return;
  }

  // 2. Load model
  document.getElementById('status').textContent = 'Loading model...';
  await loadModel();
  if (!model) {
    document.getElementById('status').textContent = 'Failed to load model.';
    return;
  }
  document.getElementById('status').textContent = 'Model loaded. Starting loop...';

  // 3. Start the continuous 2-second data collection loop
  startLoop();
}

/*****************************************************
 * (Optional) A function to stop the loop gracefully
 *****************************************************/
function stopLoop() {
  console.log('Stop loop called. No further collections will start.');
  continueLoop = false;
  // Also remove the devicemotion listener if currently collecting
  window.removeEventListener('devicemotion', handleMotionEvent, true);
  document.getElementById('status').textContent = 'Continuous loop stopped.';
}
