/**
 * API Client for Sign Language Detector
 * 
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

/**
 * SignLanguageAPI - Handles API communication with the server
 */
class SignLanguageAPI {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || '';
    this.headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    this.timeout = options.timeout || 10000; // 10 second default timeout
  }
  
  /**
   * Process a frame using the server API
   * @param {string} frameData - Base64 encoded image data
   * @param {object} options - Processing options
   * @returns {Promise<object>} - Processing results
   */
  async processFrame(frameData, options = {}) {
    try {
      const requestData = {
        frame: frameData,
        options: {
          show_landmarks: options.showLandmarks !== false,
          ...options
        }
      };
      
      const response = await this.fetchWithTimeout('/process_client_frame', {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(requestData)
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Get model information
   * @param {string} modelName - Name of the model
   * @returns {Promise<object>} - Model information
   */
  async getModelInfo(modelName) {
    try {
      // Try to get detailed JSON info first
      try {
        const response = await this.fetchWithTimeout(`/models/${modelName.replace('.p', '.json')}`);
        
        if (response.ok) {
          return await response.json();
        }
      } catch (err) {
        // Ignore error and try simplified endpoint
      }
      
      // Fall back to simplified info
      const response = await this.fetchWithTimeout(`/simplified_model_info/${modelName}`);
      
      if (!response.ok) {
        throw new Error(`Model info not available: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Select a model for use
   * @param {string} modelName - Name of the model to select
   * @returns {Promise<object>} - Selection result
   */
  async selectModel(modelName) {
    try {
      const response = await this.fetchWithTimeout('/select_model', {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ model_name: modelName })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to select model: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Reload the current model
   * @returns {Promise<object>} - Reload result
   */
  async reloadModel() {
    try {
      const response = await this.fetchWithTimeout('/reload_model', {
        method: 'POST',
        headers: this.headers
      });
      
      if (!response.ok) {
        throw new Error(`Failed to reload model: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Get reference image for a letter
   * @param {string} letter - The letter to get an image for
   * @returns {Promise<object>} - Image data
   */
  async getAnswerImage(letter) {
    try {
      const response = await this.fetchWithTimeout(`/get_answer_image/${letter}`);
      
      if (!response.ok) {
        throw new Error(`Failed to get answer image: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Fetch with timeout
   * @param {string} url - URL to fetch
   * @param {object} options - Fetch options
   * @returns {Promise<Response>} - Fetch response
   */
  fetchWithTimeout(url, options = {}) {
    const fullUrl = this.baseUrl + url;
    
    return new Promise((resolve, reject) => {
      // Set up timeout
      const timeoutId = setTimeout(() => {
        reject(new Error(`Request timeout for ${url}`));
      }, this.timeout);
      
      fetch(fullUrl, options)
        .then(response => {
          clearTimeout(timeoutId);
          resolve(response);
        })
        .catch(error => {
          clearTimeout(timeoutId);
          reject(error);
        });
    });
  }
  
  /**
   * Get current predictions
   * @returns {Promise<object>} - Current predictions
   */
  async getPredictions() {
    try {
      const response = await this.fetchWithTimeout('/predictions');
      
      if (!response.ok) {
        throw new Error(`Failed to get predictions: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Get list of available models
   * @returns {Promise<Array<string>>} - List of model names
   */
  async getAvailableModels() {
    try {
      const response = await this.fetchWithTimeout('/models');
      
      if (!response.ok) {
        throw new Error(`Failed to get models: ${response.status}`);
      }
      
      const data = await response.json();
      return data.models || [];
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Get application status
   * @returns {Promise<object>} - Application status
   */
  async getStatus() {
    try {
      const response = await this.fetchWithTimeout('/status');
      
      if (!response.ok) {
        throw new Error(`Failed to get status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  /**
   * Get quiz settings
   * @returns {Promise<object>} - Quiz settings
   */
  async getQuizSettings() {
    try {
      const response = await this.fetchWithTimeout('/quiz/settings');
      
      if (!response.ok) {
        throw new Error(`Failed to get quiz settings: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
}

// Create and export global instance
window.api = new SignLanguageAPI();