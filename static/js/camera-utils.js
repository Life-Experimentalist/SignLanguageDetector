/**
 * Camera Utilities for Sign Language Detector
 *
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

// Global variables
const SignLanguageApp = (window.SignLanguageApp = window.SignLanguageApp || {
  stream: null,
  isStreaming: false,
  animationId: null,
  ctx: null,
  videoCanvas: null,
  webcam: null,
  showLandmarks: true,
});

/**
 * Start camera stream
 * @param {string} deviceId - Camera device ID
 * @returns {Promise<boolean>} - Success status
 */
async function startCamera(deviceId) {
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    };

    // If a specific camera is selected
    if (deviceId) {
      constraints.video.deviceId = { exact: deviceId };
    }

    // Stop any existing stream
    if (SignLanguageApp.stream) {
      SignLanguageApp.stream.getTracks().forEach((track) => track.stop());
    }

    // Start new stream
    SignLanguageApp.stream =
      await navigator.mediaDevices.getUserMedia(constraints);
    SignLanguageApp.webcam = document.getElementById("webcam");

    if (SignLanguageApp.webcam) {
      SignLanguageApp.webcam.srcObject = SignLanguageApp.stream;

      // Wait for metadata to load
      return new Promise((resolve) => {
        SignLanguageApp.webcam.onloadedmetadata = () => {
          SignLanguageApp.isStreaming = true;
          resolve(true);
        };
        SignLanguageApp.webcam.onerror = () => {
          SignLanguageApp.isStreaming = false;
          resolve(false);
        };
      });
    }

    return false;
  } catch (err) {
    console.error(`Camera error: ${err.message}`);
    return false;
  }
}

/**
 * Stop camera stream
 * @returns {boolean} - Success status
 */
function stopCamera() {
  if (SignLanguageApp.stream) {
    SignLanguageApp.stream.getTracks().forEach((track) => track.stop());
    SignLanguageApp.isStreaming = false;

    if (SignLanguageApp.animationId) {
      cancelAnimationFrame(SignLanguageApp.animationId);
      SignLanguageApp.animationId = null;
    }

    if (SignLanguageApp.ctx && SignLanguageApp.videoCanvas) {
      SignLanguageApp.ctx.clearRect(
        0,
        0,
        SignLanguageApp.videoCanvas.width,
        SignLanguageApp.videoCanvas.height,
      );
    }

    return true;
  }
  return false;
}

/**
 * Process a single frame - capture from webcam and send to server
 * @returns {Promise<void>}
 */
async function processFrame() {
  if (
    !SignLanguageApp.isStreaming ||
    !SignLanguageApp.ctx ||
    !SignLanguageApp.videoCanvas ||
    !SignLanguageApp.webcam
  ) {
    return;
  }

  try {
    // Draw current webcam frame to canvas
    SignLanguageApp.ctx.drawImage(
      SignLanguageApp.webcam,
      0,
      0,
      SignLanguageApp.videoCanvas.width,
      SignLanguageApp.videoCanvas.height,
    );

    // Get canvas data as base64 image
    const frameData = SignLanguageApp.videoCanvas.toDataURL("image/jpeg", 0.8);

    // Prepare request data
    const requestData = {
      frame: frameData,
      options: {
        showLandmarks: SignLanguageApp.showLandmarks,
        landmarkStyle: "default",
      },
    };

    // Send to server for processing
    const response = await fetch("/process_client_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Display the processed image
    const processedImage = new Image();
    processedImage.onload = () => {
      // Clear canvas before drawing new frame
      SignLanguageApp.ctx.clearRect(
        0,
        0,
        SignLanguageApp.videoCanvas.width,
        SignLanguageApp.videoCanvas.height,
      );

      // Draw the processed image
      SignLanguageApp.ctx.drawImage(
        processedImage,
        0,
        0,
        SignLanguageApp.videoCanvas.width,
        SignLanguageApp.videoCanvas.height,
      );

      // Update metrics display
      const brightnessDisplay = document.getElementById("brightnessDisplay");
      const contrastDisplay = document.getElementById("contrastDisplay");

      if (brightnessDisplay) {
        if (data.low_brightness) {
          brightnessDisplay.textContent = `Brightness: ${data.brightness.toFixed(
            2,
          )}`;
          brightnessDisplay.style.display = "block";
        } else {
          brightnessDisplay.style.display = "none";
        }
      }

      if (contrastDisplay) {
        contrastDisplay.textContent = `Contrast: ${data.contrast.toFixed(2)}`;
        contrastDisplay.style.display = data.contrast ? "block" : "none";
      }

      // Dispatch prediction event
      const predictionEvent = new CustomEvent("signPrediction", {
        detail: {
          prediction: data.prediction,
          brightness: data.brightness,
          contrast: data.contrast,
          low_brightness: data.low_brightness,
          confidence: data.confidence || 0,
        },
      });
      document.dispatchEvent(predictionEvent);

      // Continue processing frames
      SignLanguageApp.animationId = requestAnimationFrame(processFrame);
    };

    processedImage.onerror = () => {
      console.error("Error loading processed image");

      // Continue processing with a delay
      setTimeout(() => {
        SignLanguageApp.animationId = requestAnimationFrame(processFrame);
      }, 1000);
    };

    processedImage.src = data.processed_frame;
  } catch (err) {
    console.error("Frame processing error:", err);

    // Dispatch error event
    const errorEvent = new CustomEvent("signPredictionError", {
      detail: { error: err.message },
    });
    document.dispatchEvent(errorEvent);

    // Continue processing with a delay
    setTimeout(() => {
      SignLanguageApp.animationId = requestAnimationFrame(processFrame);
    }, 1000);
  }
}

/**
 * Load available cameras into a select element
 * @param {HTMLSelectElement} selectElement - Select element to populate
 * @returns {Promise<boolean>} - Whether cameras were found
 */
async function loadCameras(selectElement) {
  try {
    if (!selectElement) {
      throw new Error("Select element not provided");
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(
      (device) => device.kind === "videoinput",
    );

    // Clear existing options
    selectElement.innerHTML = '<option value="">Select Camera</option>';

    if (videoDevices.length === 0) {
      throw new Error("No cameras found");
    }

    // Add options for each camera
    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      selectElement.appendChild(option);
    });

    // Select first camera by default if only one is available
    if (videoDevices.length === 1) {
      selectElement.selectedIndex = 1;
    }

    return true;
  } catch (err) {
    console.error(`Error loading cameras: ${err.message}`);
    return false;
  }
}

/**
 * Reload the model on the server
 * @returns {Promise<boolean>} - Whether reload was successful
 */
async function reloadModel() {
  try {
    const response = await fetch("/reload_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await response.json();

    if (data.status !== "success") {
      throw new Error(data.message || "Failed to reload model");
    }

    return true;
  } catch (err) {
    console.error(`Error reloading model: ${err.message}`);
    return false;
  }
}

/**
 * Take a screenshot from the video canvas
 * @returns {string|null} - Base64 encoded image data or null if failed
 */
function takeScreenshot() {
  try {
    if (!SignLanguageApp.videoCanvas) {
      throw new Error("Video canvas not available");
    }

    return SignLanguageApp.videoCanvas.toDataURL("image/png");
  } catch (err) {
    console.error(`Error taking screenshot: ${err.message}`);
    return null;
  }
}

// Export functions to global scope
window.startCamera = startCamera;
window.stopCamera = stopCamera;
window.processFrame = processFrame;
window.loadCameras = loadCameras;
window.reloadModel = reloadModel;
window.takeScreenshot = takeScreenshot;

// Initialize when document is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Find webcam and canvas elements
  SignLanguageApp.webcam = document.getElementById("webcam");
  SignLanguageApp.videoCanvas = document.getElementById("videoCanvas");

  if (SignLanguageApp.videoCanvas) {
    SignLanguageApp.ctx = SignLanguageApp.videoCanvas.getContext("2d");
  }

  // Initialize UI components from sidebar
  const showLandmarksCheckbox = document.getElementById("showLandmarksSidebar");
  if (showLandmarksCheckbox) {
    showLandmarksCheckbox.addEventListener("change", () => {
      SignLanguageApp.showLandmarks = showLandmarksCheckbox.checked;
    });

    // Set initial state
    SignLanguageApp.showLandmarks = showLandmarksCheckbox.checked;
  }
});
