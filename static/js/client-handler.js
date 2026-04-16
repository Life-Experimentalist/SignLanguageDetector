/**
 * Client Handler for Sign Language Detector
 * Handles client-side operations for multi-client implementation
 * 
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0
 */

// Create Client Handler namespace
const ClientHandler = {
    // Client state
    state: {
        isStreaming: false,
        stream: null,
        animationId: null,
        processingFrame: false,
        connectionErrors: 0,
        maxErrors: 5,
        reconnectDelay: 1000,
        lastActivityTime: Date.now(),
        // Performance tracking
        frameRate: {
            lastFrameTime: 0,
            frames: 0,
            fps: 0
        },
        // Landmark display options
        landmarkOptions: {
            showLandmarks: true,
            landmarkStyle: "default" // Options: "default", "custom", "none"
        }
    },
    
    // DOM elements
    elements: {
        webcam: null,
        canvas: null,
        ctx: null,
        predictionDisplay: null,
        errorDisplay: null,
        fpsCounter: null,
        landmarkToggle: null,
        landmarkStyleSelector: null
    },
    
    /**
     * Initialize the client handler
     * @param {Object} options - Configuration options
     */
    initialize: function(options = {}) {
        this.elements.webcam = document.getElementById(options.webcamId || "webcam");
        this.elements.canvas = document.getElementById(options.canvasId || "videoCanvas");
        
        if (this.elements.canvas) {
            this.elements.ctx = this.elements.canvas.getContext("2d");
        }
        
        this.elements.predictionDisplay = document.getElementById(options.predictionId || "predictionDisplay");
        this.elements.errorDisplay = document.getElementById(options.errorId || "errorDisplay");
        this.elements.fpsCounter = document.getElementById(options.fpsId || "fpsCounter");
        this.elements.landmarkToggle = document.getElementById(options.landmarkToggleId || "landmarkToggle");
        this.elements.landmarkStyleSelector = document.getElementById(options.landmarkStyleId || "landmarkStyle");
        
        // Setup landmark toggle if element exists
        if (this.elements.landmarkToggle) {
            this.elements.landmarkToggle.checked = this.state.landmarkOptions.showLandmarks;
            this.elements.landmarkToggle.addEventListener("change", (e) => {
                this.state.landmarkOptions.showLandmarks = e.target.checked;
                
                // Update landmark style selector visibility
                if (this.elements.landmarkStyleSelector) {
                    this.elements.landmarkStyleSelector.disabled = !e.target.checked;
                }
            });
        }
        
        // Setup landmark style selector if element exists
        if (this.elements.landmarkStyleSelector) {
            this.elements.landmarkStyleSelector.value = this.state.landmarkOptions.landmarkStyle;
            this.elements.landmarkStyleSelector.disabled = !this.state.landmarkOptions.showLandmarks;
            this.elements.landmarkStyleSelector.addEventListener("change", (e) => {
                this.state.landmarkOptions.landmarkStyle = e.target.value;
            });
        }
        
        // Setup activity tracking
        document.addEventListener("mousemove", this._updateActivity.bind(this));
        document.addEventListener("keypress", this._updateActivity.bind(this));
        document.addEventListener("click", this._updateActivity.bind(this));
        
        // Create FPS counter if doesn't exist but enabled
        if (options.showFps && !this.elements.fpsCounter) {
            this.elements.fpsCounter = document.createElement("div");
            this.elements.fpsCounter.id = "fpsCounter";
            this.elements.fpsCounter.className = "fps-counter";
            this.elements.fpsCounter.style = "position: fixed; bottom: 10px; left: 10px; background-color: rgba(0,0,0,0.5); color: white; padding: 5px; border-radius: 3px; font-size: 12px;";
            document.body.appendChild(this.elements.fpsCounter);
        }
        
        // Setup FPS calculation
        setInterval(() => {
            const now = performance.now();
            const elapsed = now - this.state.frameRate.lastFrameTime;
            
            if (elapsed >= 1000) { // Update every second
                this.state.frameRate.fps = Math.round((this.state.frameRate.frames * 1000) / elapsed);
                this.state.frameRate.frames = 0;
                this.state.frameRate.lastFrameTime = now;
                
                if (this.elements.fpsCounter) {
                    this.elements.fpsCounter.textContent = `${this.state.frameRate.fps} FPS`;
                }
            }
        }, 500);
        
        console.log("ClientHandler initialized");
        return this;
    },
    
    /**
     * Update last activity time
     * @private
     */
    _updateActivity: function() {
        this.state.lastActivityTime = Date.now();
    },
    
    /**
     * Get available cameras
     * @param {HTMLSelectElement} selectElement - Select element to populate with cameras
     * @returns {Promise<boolean>} - True if cameras were found
     */
    loadCameras: async function(selectElement) {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            if (!selectElement) {
                return videoDevices.length > 0;
            }
            
            // Clear select element
            selectElement.innerHTML = '<option value="">Select Camera</option>';
            
            // Add options for each camera
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${index + 1}`;
                selectElement.appendChild(option);
            });
            
            // Select first camera if only one
            if (videoDevices.length === 1) {
                selectElement.value = videoDevices[0].deviceId;
            }
            
            return videoDevices.length > 0;
        } catch (err) {
            this.handleError(`Error loading cameras: ${err.message}`);
            return false;
        }
    },
    
    /**
     * Start camera with specified device ID
     * @param {string} deviceId - Camera device ID
     * @returns {Promise<boolean>} - True if camera started successfully
     */
    startCamera: async function(deviceId) {
        try {
            // Clear any existing error
            if (this.elements.errorDisplay) {
                this.elements.errorDisplay.textContent = "";
            }
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            };
            
            if (deviceId) {
                constraints.video.deviceId = { exact: deviceId };
            }
            
            // Stop any existing stream
            if (this.state.stream) {
                this.stopCamera();
            }
            
            // Start new stream
            this.state.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.elements.webcam.srcObject = this.state.stream;
            
            // Wait for video to be ready
            return new Promise((resolve) => {
                this.elements.webcam.onloadedmetadata = () => {
                    this.state.isStreaming = true;
                    this.state.connectionErrors = 0;
                    resolve(true);
                };
                
                this.elements.webcam.onerror = () => {
                    this.handleError("Error starting video stream");
                    resolve(false);
                };
            });
        } catch (err) {
            this.handleError(`Camera error: ${err.message}`);
            return false;
        }
    },
    
    /**
     * Stop camera and clean up
     * @returns {boolean} - True if camera was stopped
     */
    stopCamera: function() {
        if (this.state.stream) {
            this.state.stream.getTracks().forEach(track => track.stop());
            this.state.isStreaming = false;
            
            // Cancel animation frame
            if (this.state.animationId) {
                cancelAnimationFrame(this.state.animationId);
                this.state.animationId = null;
            }
            
            // Clear canvas
            if (this.elements.ctx && this.elements.canvas) {
                this.elements.ctx.clearRect(0, 0, this.elements.canvas.width, this.elements.canvas.height);
            }
            
            return true;
        }
        return false;
    },
    
    /**
     * Process a single frame
     * @returns {void}
     */
    processFrame: async function() {
        // Don't process if not streaming or already processing a frame
        if (!this.state.isStreaming || this.state.processingFrame || 
            !this.elements.ctx || !this.elements.canvas || !this.elements.webcam) {
            this.state.animationId = requestAnimationFrame(this.processFrame.bind(this));
            return;
        }
        
        try {
            this.state.processingFrame = true;
            
            // Draw current webcam frame to canvas
            this.elements.ctx.drawImage(
                this.elements.webcam, 
                0, 0, 
                this.elements.canvas.width, 
                this.elements.canvas.height
            );
            
            // Convert canvas to base64 image (use JPEG for better performance)
            const frameData = this.elements.canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for processing with landmark options
            const response = await fetch("/process_client_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    frame: frameData,
                    options: {
                        showLandmarks: this.state.landmarkOptions.showLandmarks,
                        landmarkStyle: this.state.landmarkOptions.landmarkStyle
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Reset connection errors on success
            this.state.connectionErrors = 0;
            
            // Display processed image (with or without landmarks based on settings)
            const processedImage = new Image();
            processedImage.onload = () => {
                if (!this.state.isStreaming) return;
                
                // Clear canvas
                this.elements.ctx.clearRect(0, 0, this.elements.canvas.width, this.elements.canvas.height);
                
                // If landmarks are turned off, draw the original webcam feed
                // Otherwise, draw the processed image with landmarks
                if (!this.state.landmarkOptions.showLandmarks && data.original_frame) {
                    const originalImage = new Image();
                    originalImage.onload = () => {
                        this.elements.ctx.drawImage(
                            originalImage, 
                            0, 0, 
                            this.elements.canvas.width, 
                            this.elements.canvas.height
                        );
                    };
                    originalImage.src = data.original_frame;
                } else {
                    this.elements.ctx.drawImage(
                        processedImage, 
                        0, 0, 
                        this.elements.canvas.width, 
                        this.elements.canvas.height
                    );
                }
                
                // Dispatch prediction event
                const predictionEvent = new CustomEvent("signPrediction", { 
                    detail: { 
                        prediction: data.prediction,
                        brightness: data.brightness,
                        contrast: data.contrast,
                        lowBrightness: data.low_brightness
                    } 
                });
                document.dispatchEvent(predictionEvent);
                
                // Update prediction display
                if (this.elements.predictionDisplay) {
                    this.elements.predictionDisplay.textContent = data.prediction || "No sign detected";
                    
                    // Add low brightness warning
                    if (data.low_brightness) {
                        this.elements.predictionDisplay.classList.add("low-brightness");
                    } else {
                        this.elements.predictionDisplay.classList.remove("low-brightness");
                    }
                }
                
                // Count this frame for FPS calculation
                this.state.frameRate.frames++;
                
                // Release lock and request next frame
                this.state.processingFrame = false;
                this.state.animationId = requestAnimationFrame(this.processFrame.bind(this));
            };
            
            processedImage.src = data.processed_frame;
            
        } catch (error) {
            console.error("Frame processing error:", error);
            this.state.connectionErrors++;
            
            // Handle reconnection logic
            if (this.state.connectionErrors >= this.state.maxErrors) {
                this.handleError(`Connection lost: ${error.message}. Reconnecting...`);
                
                // Attempt reconnection after delay
                setTimeout(() => {
                    this.state.connectionErrors = 0;
                    this.state.processingFrame = false;
                    this.state.animationId = requestAnimationFrame(this.processFrame.bind(this));
                }, this.state.reconnectDelay);
                
                return;
            }
            
            // For non-critical errors, continue processing
            this.state.processingFrame = false;
            this.state.animationId = requestAnimationFrame(this.processFrame.bind(this));
        }
    },
    
    /**
     * Start continuous frame processing
     * @returns {void}
     */
    startProcessing: function() {
        if (!this.state.isStreaming) {
            console.warn("Cannot start processing: Camera not streaming");
            return false;
        }
        
        this.state.processingFrame = false;
        this.state.animationId = requestAnimationFrame(this.processFrame.bind(this));
        return true;
    },
    
    /**
     * Stop processing frames
     * @returns {void}
     */
    stopProcessing: function() {
        if (this.state.animationId) {
            cancelAnimationFrame(this.state.animationId);
            this.state.animationId = null;
            this.state.processingFrame = false;
            return true;
        }
        return false;
    },
    
    /**
     * Handle and display errors
     * @param {string} message - Error message
     */
    handleError: function(message) {
        console.error(message);
        
        if (this.elements.errorDisplay) {
            this.elements.errorDisplay.textContent = message;
            this.elements.errorDisplay.style.display = "block";
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                if (this.elements.errorDisplay.textContent === message) {
                    this.elements.errorDisplay.style.display = "none";
                }
            }, 5000);
        }
    },
    
    /**
     * Take a still image from the current camera
     * @returns {Promise<string>} Base64 encoded image data
     */
    takeSnapshot: async function() {
        if (!this.state.isStreaming || !this.elements.webcam) {
            throw new Error("Camera not streaming");
        }
        
        // Create temporary canvas if needed
        const canvas = this.elements.canvas || document.createElement('canvas');
        const ctx = this.elements.ctx || canvas.getContext('2d');
        
        // Set canvas dimensions to match video
        canvas.width = this.elements.webcam.videoWidth;
        canvas.height = this.elements.webcam.videoHeight;
        
        // Draw video frame to canvas
        ctx.drawImage(this.elements.webcam, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        return canvas.toDataURL('image/jpeg');
    },
    
    /**
     * Check if the camera has been inactive for a specified period
     * @param {number} inactivityThreshold - Inactivity threshold in milliseconds
     * @returns {boolean} True if inactive
     */
    checkInactivity: function(inactivityThreshold = 300000) { // 5 minutes
        const elapsed = Date.now() - this.state.lastActivityTime;
        return elapsed > inactivityThreshold;
    },
    
    /**
     * Clean up resources
     * @returns {void}
     */
    cleanup: function() {
        this.stopCamera();
        
        // Remove event listeners
        document.removeEventListener("mousemove", this._updateActivity);
        document.removeEventListener("keypress", this._updateActivity);
        document.removeEventListener("click", this._updateActivity);
    },
    
    /**
     * Set landmark display options
     * @param {boolean} show - Whether to show landmarks
     * @param {string} style - Landmark style ("default", "custom", "none")
     */
    setLandmarkOptions: function(show, style = "default") {
        this.state.landmarkOptions.showLandmarks = show;
        this.state.landmarkOptions.landmarkStyle = style;
        
        // Update UI elements if they exist
        if (this.elements.landmarkToggle) {
            this.elements.landmarkToggle.checked = show;
        }
        
        if (this.elements.landmarkStyleSelector) {
            this.elements.landmarkStyleSelector.value = style;
            this.elements.landmarkStyleSelector.disabled = !show;
        }
    }
};

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClientHandler;
}