/**
 * Model Selection and Management
 *
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

/**
 * Load model information into the UI
 * @param {string} modelName - Name of the model to load
 * @returns {Promise<boolean>} - Whether load was successful
 */
async function loadModelInfo(modelName) {
  if (!modelName) return false;

  try {
    // Try to load from JSON file first
    const response = await fetch(`/models/${modelName.replace(".p", ".json")}`);

    if (!response.ok) {
      // Fall back to simplified API
      return await loadSimplifiedModelInfo(modelName);
    }

    const data = await response.json();
    updateModelInfoUI(data);
    return true;
  } catch (error) {
    console.error("Error loading model info:", error);
    return false;
  }
}

/**
 * Load simplified model information from API endpoint
 * @param {string} modelName - Name of the model to load
 * @returns {Promise<boolean>} - Whether load was successful
 */
async function loadSimplifiedModelInfo(modelName) {
  try {
    const response = await fetch(`/simplified_model_info/${modelName}`);

    if (!response.ok) {
      throw new Error("Model info not available");
    }

    const data = await response.json();
    updateSimplifiedInfoUI(data);
    return true;
  } catch (error) {
    console.error("Error loading simplified model info:", error);
    return false;
  }
}

/**
 * Update UI with model information
 * @param {object} data - Model information data
 */
function updateModelInfoUI(data) {
  // Check if we have the floating card for model info
  const modelInfoCard = document.querySelector(".model-info");
  if (!modelInfoCard) return;

  // Extract model information
  const modelType = data.model_info.type || "Unknown";
  const accuracy = data.performance.accuracy || 0;
  const formattedAccuracy = `${(accuracy * 100).toFixed(1)}%`;

  // Get class information
  const classes = data.performance.classes || {};

  // Sort classes by F1 score
  const sortedClasses = Object.entries(classes)
    .filter(([className]) => !isNaN(className)) // Only numeric class names
    .map(([className, metrics]) => ({
      className,
      metrics,
    }))
    .sort((a, b) => (b.metrics.f1 || 0) - (a.metrics.f1 || 0));

  // Update UI
  modelInfoCard.innerHTML = `
    <div class="accuracy-display">
      <div class="accuracy-badge">${formattedAccuracy}</div>
    </div>
    
    <div class="model-title">
      <strong>Model:</strong> ${modelType}
    </div>
    
    <div class="top-classes">
      <h3>Top Performing Classes</h3>
      ${sortedClasses
        .slice(0, 5)
        .map(
          (classInfo, index) => `
        <div class="top-class-item">
          <div class="class-label">
            <span class="class-badge">${
              parseInt(classInfo.className) + 1
            }</span>
            ${getLabelForClass(parseInt(classInfo.className))}
          </div>
          <div class="class-value">${(classInfo.metrics.f1 * 100).toFixed(
            1
          )}%</div>
        </div>
      `
        )
        .join("")}
    </div>
  `;
}

/**
 * Update UI with simplified model information
 * @param {object} data - Simplified model information data
 */
function updateSimplifiedInfoUI(data) {
  // Check if we have the floating card for model info
  const modelInfoCard = document.querySelector(".model-info");
  if (!modelInfoCard) return;

  // Extract available information
  const highlights = data.highlights || {};
  const accuracy = highlights.accuracy || { formatted: "N/A" };
  const topClasses = highlights.top_classes || [];

  // Update UI
  modelInfoCard.innerHTML = `
    <div class="accuracy-display">
      <div class="accuracy-badge">${accuracy.formatted}</div>
    </div>
    
    <div class="model-title">
      <strong>Model Info:</strong> Basic metrics only
    </div>
    
    <div class="top-classes">
      <h3>Top Performing Classes</h3>
      ${topClasses
        .map(
          (classInfo, index) => `
        <div class="top-class-item">
          <div class="class-label">
            <span class="class-badge">${index + 1}</span>
            ${classInfo.label}
          </div>
          <div class="class-value">${classInfo.formatted}</div>
        </div>
      `
        )
        .join("")}
    </div>
  `;
}

/**
 * Get human-readable label for class index
 * @param {number} classIndex - Class index
 * @returns {string} - Human-readable label
 */
function getLabelForClass(classIndex) {
  // Convert numeric class to letter (0 = A, 1 = B, etc.)
  if (classIndex >= 0 && classIndex <= 25) {
    return String.fromCharCode(65 + classIndex);
  }
  return `Class ${classIndex}`;
}

/**
 * Select a model and notify the server
 * @param {string} modelName - Name of the model to select
 * @returns {Promise<boolean>} - Whether selection was successful
 */
async function selectModel(modelName) {
  try {
    const response = await fetch("/select_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName }),
    });

    const data = await response.json();

    if (data.status !== "success") {
      throw new Error(data.message || "Failed to select model");
    }

    // Load model info after selection
    await loadModelInfo(modelName);

    // Update sidebar if available
    const currentModelElement = document.getElementById("currentModelName");
    if (currentModelElement) {
      currentModelElement.textContent = modelName;
    }

    // Use loadModelInfoSidebar if available (from sidebar.html)
    if (typeof loadModelInfoSidebar === "function") {
      loadModelInfoSidebar(modelName);
    }

    return true;
  } catch (error) {
    console.error("Error selecting model:", error);
    return false;
  }
}

// Initialize model selection when the document is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Get model select dropdown
  const modelSelect = document.getElementById("modelSelect");

  if (modelSelect) {
    // Load current model info
    const selectedModel = modelSelect.value;
    if (selectedModel) {
      loadModelInfo(selectedModel);
    }

    // Handle model selection changes
    modelSelect.addEventListener("change", async () => {
      const modelName = modelSelect.value;

      if (!modelName) return;

      try {
        const success = await selectModel(modelName);

        if (success && window.showCustomAlert) {
          window.showCustomAlert(`Model switched to ${modelName}`);
        }
      } catch (error) {
        console.error("Model selection error:", error);

        if (window.showCustomAlert) {
          window.showCustomAlert(`Error selecting model: ${error.message}`);
        }
      }
    });
  }
});

// Export functions to global scope
window.loadModelInfo = loadModelInfo;
window.selectModel = selectModel;
