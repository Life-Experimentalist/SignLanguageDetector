/**
 * Debug Tools for Sign Language Detector
 *
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

/**
 * Debug Tools - Utilities for debugging and troubleshooting
 */
class DebugTools {
  constructor(options = {}) {
    this.enabled = options.enabled || false;
    this.container = null;
    this.logElement = null;
    this.statsElement = null;
    this.eventListeners = [];
    
    // Initialize if enabled
    if (this.enabled) {
      this.init();
    }
  }
  
  /**
   * Initialize the debug tools interface
   */
  init() {
    // Create debug panel
    this.createDebugPanel();
    
    // Listen for prediction events
    this.addEventListener(document, 'signPrediction', this.onPrediction.bind(this));
    this.addEventListener(document, 'signPredictionError', this.onPredictionError.bind(this));
    
    // Start performance monitoring
    this.startPerformanceMonitoring();
    
    console.log('Debug tools initialized');
  }
  
  /**
   * Create the debug panel interface
   */
  createDebugPanel() {
    // Create container element
    this.container = document.createElement('div');
    this.container.className = 'debug-panel';
    this.container.style.position = 'fixed';
    this.container.style.bottom = '10px';
    this.container.style.left = '10px';
    this.container.style.width = '300px';
    this.container.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    this.container.style.color = '#fff';
    this.container.style.padding = '10px';
    this.container.style.borderRadius = '5px';
    this.container.style.fontFamily = 'monospace';
    this.container.style.fontSize = '12px';
    this.container.style.zIndex = '9999';
    this.container.style.transition = 'transform 0.3s';
    this.container.style.transform = 'translateY(90%)';
    this.container.style.backdropFilter = 'blur(5px)';
    
    // Create header
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.marginBottom = '10px';
    header.style.cursor = 'pointer';
    header.innerHTML = '<strong>Debug Tools</strong>';
    header.addEventListener('click', () => {
      if (this.container.style.transform === 'translateY(0px)') {
        this.container.style.transform = 'translateY(90%)';
      } else {
        this.container.style.transform = 'translateY(0px)';
      }
    });
    
    // Create close button
    const closeButton = document.createElement('button');
    closeButton.innerHTML = '×';
    closeButton.style.background = 'transparent';
    closeButton.style.border = 'none';
    closeButton.style.color = '#fff';
    closeButton.style.fontSize = '16px';
    closeButton.style.cursor = 'pointer';
    closeButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.container.remove();
      this.enabled = false;
    });
    
    header.appendChild(closeButton);
    this.container.appendChild(header);
    
    // Create tabs
    const tabs = document.createElement('div');
    tabs.className = 'debug-tabs';
    tabs.style.display = 'flex';
    tabs.style.marginBottom = '10px';
    
    const tabLog = document.createElement('div');
    tabLog.textContent = 'Log';
    tabLog.style.padding = '5px 10px';
    tabLog.style.backgroundColor = '#333';
    tabLog.style.borderTopLeftRadius = '3px';
    tabLog.style.borderBottomLeftRadius = '3px';
    tabLog.style.cursor = 'pointer';
    
    const tabStats = document.createElement('div');
    tabStats.textContent = 'Stats';
    tabStats.style.padding = '5px 10px';
    tabStats.style.backgroundColor = '#555';
    tabStats.style.borderTopRightRadius = '3px';
    tabStats.style.borderBottomRightRadius = '3px';
    tabStats.style.cursor = 'pointer';
    
    tabs.appendChild(tabLog);
    tabs.appendChild(tabStats);
    this.container.appendChild(tabs);
    
    // Create log element
    this.logElement = document.createElement('div');
    this.logElement.className = 'debug-log';
    this.logElement.style.height = '150px';
    this.logElement.style.overflowY = 'auto';
    this.logElement.style.wordBreak = 'break-all';
    this.container.appendChild(this.logElement);
    
    // Create stats element
    this.statsElement = document.createElement('div');
    this.statsElement.className = 'debug-stats';
    this.statsElement.style.height = '150px';
    this.statsElement.style.overflowY = 'auto';
    this.statsElement.style.display = 'none';
    this.container.appendChild(this.statsElement);
    
    // Tab switching
    tabLog.addEventListener('click', () => {
      this.logElement.style.display = 'block';
      this.statsElement.style.display = 'none';
      tabLog.style.backgroundColor = '#333';
      tabStats.style.backgroundColor = '#555';
    });
    
    tabStats.addEventListener('click', () => {
      this.logElement.style.display = 'none';
      this.statsElement.style.display = 'block';
      tabLog.style.backgroundColor = '#555';
      tabStats.style.backgroundColor = '#333';
    });
    
    // Create debug controls
    const controls = document.createElement('div');
    controls.className = 'debug-controls';
    controls.style.display = 'flex';
    controls.style.justifyContent = 'space-between';
    controls.style.marginTop = '10px';
    
    // Create clear button
    const clearButton = document.createElement('button');
    clearButton.textContent = 'Clear Log';
    clearButton.style.padding = '3px 8px';
    clearButton.style.backgroundColor = '#333';
    clearButton.style.color = '#fff';
    clearButton.style.border = 'none';
    clearButton.style.borderRadius = '3px';
    clearButton.style.cursor = 'pointer';
    clearButton.addEventListener('click', () => {
      this.clearLog();
    });
    
    // Create reload button
    const reloadButton = document.createElement('button');
    reloadButton.textContent = 'Reload Model';
    reloadButton.style.padding = '3px 8px';
    reloadButton.style.backgroundColor = '#007bff';
    reloadButton.style.color = '#fff';
    reloadButton.style.border = 'none';
    reloadButton.style.borderRadius = '3px';
    reloadButton.style.cursor = 'pointer';
    reloadButton.addEventListener('click', () => {
      this.log('Reloading model...');
      window.api.reloadModel()
        .then(data => this.log(`Model reloaded: ${JSON.stringify(data)}`))
        .catch(error => this.log(`Reload error: ${error.message}`, 'error'));
    });
    
    controls.appendChild(clearButton);
    controls.appendChild(reloadButton);
    this.container.appendChild(controls);
    
    // Append to body
    document.body.appendChild(this.container);
  }
  
  /**
   * Add an event listener and track it for cleanup
   * @param {EventTarget} target - Element to listen on
   * @param {string} type - Event type
   * @param {Function} listener - Event listener
   */
  addEventListener(target, type, listener) {
    target.addEventListener(type, listener);
    this.eventListeners.push({ target, type, listener });
  }
  
  /**
   * Clean up all event listeners
   */
  cleanupEventListeners() {
    this.eventListeners.forEach(({ target, type, listener }) => {
      target.removeEventListener(type, listener);
    });
    this.eventListeners = [];
  }
  
  /**
   * Start monitoring performance
   */
  startPerformanceMonitoring() {
    // Track frame rate
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let fps = 0;
    
    // Track predictions
    let predictionCount = 0;
    let lastPredictionTime = performance.now();
    let pps = 0; // predictions per second
    
    // Update stats every second
    setInterval(() => {
      const now = performance.now();
      
      // Calculate FPS
      fps = Math.round(frameCount / ((now - lastFrameTime) / 1000));
      frameCount = 0;
      lastFrameTime = now;
      
      // Calculate PPS
      pps = Math.round(predictionCount / ((now - lastPredictionTime) / 1000));
      predictionCount = 0;
      lastPredictionTime = now;
      
      // Update stats display
      this.updateStats(fps, pps);
    }, 1000);
    
    // Track frames
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    window.requestAnimationFrame = callback => {
      return originalRequestAnimationFrame(timestamp => {
        frameCount++;
        callback(timestamp);
      });
    };
    
    // Track predictions
    this.addEventListener(document, 'signPrediction', () => {
      predictionCount++;
    });
  }
  
  /**
   * Update stats display
   * @param {number} fps - Frames per second
   * @param {number} pps - Predictions per second
   */
  updateStats(fps, pps) {
    if (!this.statsElement) return;
    
    // Get memory usage if available
    let memoryStats = '';
    if (window.performance && window.performance.memory) {
      const memory = window.performance.memory;
      const usedHeapSize = Math.round(memory.usedJSHeapSize / (1024 * 1024));
      const totalHeapSize = Math.round(memory.totalJSHeapSize / (1024 * 1024));
      memoryStats = `Memory: ${usedHeapSize}MB / ${totalHeapSize}MB<br>`;
    }
    
    // Update stats display
    this.statsElement.innerHTML = `
      <div style="margin-bottom: 10px;">
        <strong>Performance</strong>
        <div>FPS: <span style="color: ${fps < 15 ? '#ff5555' : '#55ff55'}">${fps}</span></div>
        <div>Predictions/sec: <span style="color: ${pps < 5 ? '#ff5555' : '#55ff55'}">${pps}</span></div>
        ${memoryStats}
      </div>
      
      <div>
        <strong>System</strong>
        <div>Viewport: ${window.innerWidth}x${window.innerHeight}</div>
        <div>User Agent: ${navigator.userAgent.substring(0, 50)}...</div>
      </div>
    `;
  }
  
  /**
   * Log a message to the debug console
   * @param {string} message - Message to log
   * @param {string} level - Log level (info, error, warn)
   */
  log(message, level = 'info') {
    if (!this.logElement) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const logItem = document.createElement('div');
    logItem.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';
    logItem.style.padding = '3px 0';
    
    // Set color based on log level
    let color = '#fff';
    switch (level) {
      case 'error':
        color = '#ff5555';
        break;
      case 'warn':
        color = '#ffff55';
        break;
      case 'success':
        color = '#55ff55';
        break;
    }
    
    logItem.innerHTML = `
      <div style="color: #aaa; font-size: 10px;">${timestamp}</div>
      <div style="color: ${color};">${message}</div>
    `;
    
    this.logElement.appendChild(logItem);
    this.logElement.scrollTop = this.logElement.scrollHeight;
    
    // Also log to console
    console[level](message);
  }
  
  /**
   * Clear the log display
   */
  clearLog() {
    if (this.logElement) {
      this.logElement.innerHTML = '';
    }
  }
  
  /**
   * Handle prediction events
   * @param {CustomEvent} event - Prediction event
   */
  onPrediction(event) {
    const { prediction, confidence } = event.detail;
    if (prediction) {
      this.log(`Prediction: ${prediction} (${Math.round(confidence * 100)}%)`);
    }
  }
  
  /**
   * Handle prediction error events
   * @param {CustomEvent} event - Prediction error event
   */
  onPredictionError(event) {
    const { error } = event.detail;
    this.log(`Prediction error: ${error}`, 'error');
  }
  
  /**
   * Enable debug tools
   */
  enable() {
    this.enabled = true;
    this.init();
  }
  
  /**
   * Disable debug tools
   */
  disable() {
    this.enabled = false;
    this.cleanupEventListeners();
    if (this.container) {
      this.container.remove();
      this.container = null;
      this.logElement = null;
      this.statsElement = null;
    }
  }
}

// Create global debug tools instance
window.debugTools = new DebugTools({
  // Enable in development mode or when debug parameter is present
  enabled: window.location.hostname === 'localhost' || window.location.search.includes('debug=true')
});

// Add keyboard shortcut to toggle debug tools (Ctrl+Shift+D)
document.addEventListener('keydown', function(event) {
  if (event.ctrlKey && event.shiftKey && event.key === 'D') {
    event.preventDefault();
    if (window.debugTools.enabled) {
      window.debugTools.disable();
    } else {
      window.debugTools.enable();
    }
  }
});

// Override console.log to include in debug panel when enabled
const originalConsoleLog = console.log;
console.log = function(...args) {
  originalConsoleLog.apply(console, args);
  if (window.debugTools && window.debugTools.enabled) {
    window.debugTools.log(args.map(arg => {
      if (typeof arg === 'object') {
        try {
          return JSON.stringify(arg);
        } catch (e) {
          return String(arg);
        }
      }
      return String(arg);
    }).join(' '));
  }
};

// Override console.error to include in debug panel when enabled
const originalConsoleError = console.error;
console.error = function(...args) {
  originalConsoleError.apply(console, args);
  if (window.debugTools && window.debugTools.enabled) {
    window.debugTools.log(args.map(arg => String(arg)).join(' '), 'error');
  }
};
