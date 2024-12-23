/*****************************************************
 * Global Variables
 *****************************************************/
let model = null;            // Holds the loaded TF.js model
let accelData = [];          // Stores the raw accelerometer data
const TARGET_SAMPLES = 52;   // 52 samples total
const COLLECTION_TIME = 2000; // 2 seconds
let continueLoop = true;     // Control variable to stop the loop if desired
let inferenceCount = 0;

/*****************************************************
 * requestMotionPermission() 
 * (For iOS >= 13)
 *****************************************************/
async function requestMotionPermission() {
  console.log('[requestMotionPermission] Checking for iOS permission...');
  if (
    typeof DeviceMotionEvent !== 'undefined' &&
    typeof DeviceMotionEvent.requestPermission === 'function'
  ) {
    try {
      console.log('[requestMotionPermission] Requesting permission...');
      const response = await DeviceMotionEvent.requestPermission();
      if (response === 'granted') {
        console.log('[requestMotionPermission] Permission granted.');
        return true;
      } else {
        console.warn('[requestMotionPermission] Permission denied.');
        return false;
      }
    } catch (err) {
      console.error('[requestMotionPermission] Error requesting permission:', err);
      return false;
    }
  } else {
    // Non-iOS or older iOS
    console.log('[requestMotionPermission] No special permission required on this device.');
    return true;
  }
}

/*****************************************************
 * loadModel()
 *****************************************************/
async function loadModel() {
  try {
    console.log('[loadModel] Loading model from: model_web/model.json');
    model = await tf.loadLayersModel('model_web/model.json');
    console.log('[loadModel] Model loaded successfully:', model);
  } catch (error) {
    console.error('[loadModel] Error loading model:', error);
    model = null;
  }
}

/*****************************************************
 * startContinuousLoop()
 * - Kicks off the first data collection cycle
 *****************************************************/
function startContinuousLoop() {
  console.log('[startContinuousLoop] Initiating continuous capture loop...');
  continueLoop = true;
  startCollectingData();
}

/*****************************************************
 * stopContinuousLoop()
 * - Allows you to stop the loop if needed
 *****************************************************/
function stopContinuousLoop() {
  console.log('[stopContinuousLoop] Stopping continuous capture loop...');
  continueLoop = false;
  // Also remove any devicemotion listener if currently active
  window.removeEventListener('devicemotion', handleMotionEvent, true);
  document.getElementById('status').textContent = 'Loop stopped.';
}

/*****************************************************
 * startCollectingData()
 * - Collect data for 2 seconds, then stop
 *****************************************************/
function startCollectingData() {
  if (!continueLoop) {
    console.log('[startCollectingData] continueLoop = false, so not starting.');
    return;
  }

  console.log('[startCollectingData] Reset accelData and add devicemotion listener.');
  accelData = [];
  window.addEventListener('devicemotion', handleMotionEvent, true);

  // After 2 seconds, stop and run inference
  setTimeout(() => {
    stopCollectingData();
  }, COLLECTION_TIME);
}

/*****************************************************
 * stopCollectingData()
 * - Called after 2 seconds
 *****************************************************/
async function stopCollectingData() {
  console.log('[stopCollectingData] Removing devicemotion listener.');
  window.removeEventListener('devicemotion', handleMotionEvent, true);

  // Pad or truncate to EXACTLY 52 samples
  console.log(`[stopCollectingData] Collected ${accelData.length} samples.`);
  if (accelData.length < TARGET_SAMPLES) {
    while (accelData.length < TARGET_SAMPLES) {
      accelData.push({ x: 0, y: 0, z: 0 });
    }
    console.log(`[stopCollectingData] Padded to ${accelData.length} samples.`);
  } else if (accelData.length > TARGET_SAMPLES) {
    accelData = accelData.slice(-TARGET_SAMPLES);
    console.log(`[stopCollectingData] Truncated to ${accelData.length} samples.`);
  }

  document.getElementById('status').textContent =
    `Collected ${accelData.length} samples. Running inference...`;

  // Run inference
  await runInference();

  // If continueLoop is still true, start another cycle
  if (continueLoop) {
    console.log('[stopCollectingData] Starting next 2s cycle...');
    startCollectingData();
  } else {
    console.log('[stopCollectingData] continueLoop=false, so no more cycles.');
  }
}

/*****************************************************
 * handleMotionEvent()
 * - Triggered by devicemotion
 *****************************************************/
function handleMotionEvent(event) {
  const { x, y, z } = event.accelerationIncludingGravity || {};
  
  // If x,y,z are null/undefined on some devices, fallback to 0
  accelData.push({
    x: x || 0,
    y: y || 0,
    z: z || 0
  });

  document.getElementById('status').textContent =
    `Collecting... Samples so far: ${accelData.length}`;
}

/*****************************************************
 * runInference()
 * - Builds input tensor, calls model.predict
 *****************************************************/
async function runInference() {
  if (!model) {
    console.warn('[runInference] Model is not loaded. Cannot run inference.');
    document.getElementById('status').textContent = 'Model not loaded.';
    return;
  }

  // Build tensor
  const dataArray = accelData.map(d => [[d.x], [d.y], [d.z]]);
  let inputTensor = tf.tensor3d(dataArray).reshape([1, TARGET_SAMPLES, 3, 1]);

  try {
    // Predict
    const outputTensor = model.predict(inputTensor);
    const predictions = await outputTensor.data();

    outputTensor.dispose();
    inputTensor.dispose();

    // 1) Continue to show a quick status update
    document.getElementById('status').textContent = 'Inference complete!';

    // 2) Prepend the new result at the top of "logResults" 
    const logContainer = document.getElementById('logResults');
    const p = document.createElement('p');
    p.textContent = `Inference result: ${JSON.stringify(Array.from(predictions))}`;

    // Insert the new paragraph as the first child (so the newest appears at top)
    if (logContainer.firstChild) {
      logContainer.insertBefore(p, logContainer.firstChild);
    } else {
      // If no child yet, just append
      logContainer.appendChild(p);
    }

    console.log('[runInference] Predictions:', predictions);

  } catch (err) {
    console.error('[runInference] Error during model.predict():', err);
    document.getElementById('status').textContent = `Error: ${err}`;

    // Dispose on error
    if (inputTensor) {
      inputTensor.dispose();
    }
  }
}

/*****************************************************
 * initDemo()
 * - Called (once) from the page:
 *   - request permission
 *   - load model
 *   - start the continuous loop
 *****************************************************/
async function initDemo() {
  console.log('[initDemo] Called!');

  // 1) iOS motion permission
  const granted = await requestMotionPermission();
  if (!granted) {
    document.getElementById('status').textContent = 
      'Permission for motion denied.';
    console.warn('[initDemo] Motion permission denied, stopping.');
    return;
  }

  // 2) Load model
  document.getElementById('status').textContent = 'Loading model...';
  await loadModel();
  if (!model) {
    document.getElementById('status').textContent = 'Failed to load model.';
    console.error('[initDemo] Model is null after load attempt, stopping.');
    return;
  }
  
  // 3) Start continuous loop
  document.getElementById('status').textContent = 'Model loaded. Starting continuous loop...';
  startContinuousLoop();
}
