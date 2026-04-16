/**
 * Prediction Display Handler
 *
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

class PredictionDisplay {
  constructor(options = {}) {
    // Default options
    this.options = {
      element: null,
      valueElement: null,
      confidenceElement: null,
      historyLength: 5,
      confidenceThreshold: 0.5,
      stabilityCount: 3,
      ...options,
    };

    this.predictionHistory = [];
    this.lastStablePrediction = null;
    this.stabilityCounter = 0;

    // Initialize if element is provided
    if (this.options.element) {
      this.init();
    }
  }

  /**
   * Initialize the prediction display
   * @param {HTMLElement} element - Element to display predictions in
   */
  init(element = null) {
    if (element) {
      this.options.element = element;
    }

    // Create elements if they don't exist
    if (!this.options.valueElement && this.options.element) {
      this.options.valueElement = document.createElement("span");
      this.options.valueElement.classList.add("prediction-value");
      this.options.element.appendChild(this.options.valueElement);
    }

    if (!this.options.confidenceElement && this.options.element) {
      this.options.confidenceElement = document.createElement("span");
      this.options.confidenceElement.classList.add("prediction-confidence");
      this.options.element.appendChild(this.options.confidenceElement);
    }

    // Listen for prediction events
    document.addEventListener("signPrediction", (event) => {
      this.handlePrediction(event.detail);
    });
  }

  /**
   * Handle a new prediction
   * @param {object} predictionData - Prediction data
   */
  handlePrediction(predictionData) {
    const { prediction, confidence = 0 } = predictionData;

    // Add to history and keep only recent ones
    this.predictionHistory.unshift({ prediction, confidence });
    if (this.predictionHistory.length > this.options.historyLength) {
      this.predictionHistory.pop();
    }

    // Get the most common prediction in history
    const stablePrediction = this.getMostCommonPrediction();

    // Check if prediction is stable
    if (stablePrediction === this.lastStablePrediction) {
      this.stabilityCounter++;
    } else {
      this.stabilityCounter = 1;
      this.lastStablePrediction = stablePrediction;
    }

    // Update the display
    this.updateDisplay(stablePrediction, confidence);
  }

  /**
   * Get the most common prediction from history
   * @returns {string} - The most common prediction
   */
  getMostCommonPrediction() {
    const counts = {};

    this.predictionHistory.forEach((item) => {
      // Only count predictions above threshold
      if (item.confidence >= this.options.confidenceThreshold) {
        counts[item.prediction] = (counts[item.prediction] || 0) + 1;
      }
    });

    // Find the most common prediction
    let maxCount = 0;
    let maxPrediction = null;

    for (const [prediction, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        maxPrediction = prediction;
      }
    }

    return maxPrediction;
  }

  /**
   * Update the prediction display
   * @param {string} prediction - Current prediction
   * @param {number} confidence - Prediction confidence
   */
  updateDisplay(prediction, confidence) {
    if (!this.options.valueElement) return;

    // Only show stable predictions
    const isStable = this.stabilityCounter >= this.options.stabilityCount;
    const displayText = prediction && isStable ? prediction : "None";

    // Update value
    this.options.valueElement.textContent = displayText;

    // Update confidence if element exists
    if (this.options.confidenceElement) {
      const confidenceText = confidence
        ? `(${Math.round(confidence * 100)}%)`
        : "";
      this.options.confidenceElement.textContent = confidenceText;

      // Add visual indicators based on confidence
      this.options.confidenceElement.className = "prediction-confidence";
      if (confidence > 0.8) {
        this.options.confidenceElement.classList.add("high-confidence");
      } else if (confidence > 0.5) {
        this.options.confidenceElement.classList.add("medium-confidence");
      } else if (confidence > 0) {
        this.options.confidenceElement.classList.add("low-confidence");
      }
    }

    // Trigger custom event for other components
    if (isStable && prediction) {
      const stabilizedEvent = new CustomEvent("stablePrediction", {
        detail: { prediction, confidence, history: this.predictionHistory },
      });
      document.dispatchEvent(stabilizedEvent);
    }
  }

  /**
   * Clear the prediction history
   */
  clear() {
    this.predictionHistory = [];
    this.lastStablePrediction = null;
    this.stabilityCounter = 0;

    if (this.options.valueElement) {
      this.options.valueElement.textContent = "None";
    }

    if (this.options.confidenceElement) {
      this.options.confidenceElement.textContent = "";
      this.options.confidenceElement.className = "prediction-confidence";
    }
  }

  /**
   * Get the current stable prediction
   * @returns {object} - Current prediction and confidence
   */
  getCurrentPrediction() {
    const isStable = this.stabilityCounter >= this.options.stabilityCount;

    if (!isStable || !this.lastStablePrediction) {
      return { prediction: null, confidence: 0, isStable: false };
    }

    // Find the confidence for this prediction
    const matchingPredictions = this.predictionHistory.filter(
      (item) => item.prediction === this.lastStablePrediction
    );

    // Calculate average confidence
    const avgConfidence =
      matchingPredictions.reduce((sum, item) => sum + item.confidence, 0) /
      Math.max(1, matchingPredictions.length);

    return {
      prediction: this.lastStablePrediction,
      confidence: avgConfidence,
      isStable: true,
    };
  }
}

// Create global instance if needed
window.predictionDisplay = new PredictionDisplay({
  element: document.getElementById("predictionDisplay"),
});

// Export the class
window.PredictionDisplay = PredictionDisplay;
