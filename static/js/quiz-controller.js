/**
 * Quiz Controller for Sign Language Detector
 * 
 * Project: Sign Language Detector
 * Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
 * Owner: VKrishna04
 * Organization: Life-Experimentalist
 * Licensed under the Apache License, Version 2.0 (the "License")
 */

/**
 * QuizController - Manages sign language quiz functionality
 */
class QuizController {
  constructor(options = {}) {
    // Default options
    this.options = {
      duration: 3, // seconds to hold gesture
      numGuesses: 5, // guesses before completion
      reloadInterval: 0, // reload after X questions (0 = no reload)
      letters: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
      ...options
    };
    
    // Quiz state
    this.currentLetter = null;
    this.correctGuesses = 0;
    this.skippedQuestions = 0;
    this.totalQuestions = 0;
    this.lastPrediction = null;
    this.lastPredictionTime = null;
    this.predictionCount = 0;
    this.reloadCount = 0;
    this.active = false;
    
    // UI Elements
    this.letterDisplay = null;
    this.resultDisplay = null;
    this.answerImage = null;
    
    // Bind methods
    this.handlePrediction = this.handlePrediction.bind(this);
    
    // Initialize if elements are provided
    if (options.letterDisplay || options.resultDisplay) {
      this.init({
        letterDisplay: options.letterDisplay,
        resultDisplay: options.resultDisplay,
        answerImage: options.answerImage
      });
    }
  }
  
  /**
   * Initialize the quiz controller
   * @param {Object} elements - UI elements
   */
  init(elements) {
    // Store UI elements
    this.letterDisplay = elements.letterDisplay || this.letterDisplay;
    this.resultDisplay = elements.resultDisplay || this.resultDisplay;
    this.answerImage = elements.answerImage || this.answerImage;
    
    // Listen for prediction events
    document.addEventListener('signPrediction', event => {
      const { prediction } = event.detail;
      if (this.active) {
        this.handlePrediction(prediction);
      }
    });
    
    return this;
  }
  
  /**
   * Start the quiz
   * @returns {QuizController} - The quiz controller instance for chaining
   */
  start() {
    this.active = true;
    this.correctGuesses = 0;
    this.skippedQuestions = 0;
    this.totalQuestions = 0;
    this.reloadCount = 0;
    this.nextLetter();
    
    // Fire event
    this.dispatchEvent('quizStarted');
    
    return this;
  }
  
  /**
   * Stop the quiz
   * @returns {QuizController} - The quiz controller instance for chaining
   */
  stop() {
    this.active = false;
    
    // Fire event
    this.dispatchEvent('quizStopped', {
      correct: this.correctGuesses,
      skipped: this.skippedQuestions,
      total: this.totalQuestions
    });
    
    return this;
  }
  
  /**
   * Move to the next letter
   * @returns {Promise<string>} - The next letter
   */
  async nextLetter() {
    // Select a random letter
    const letters = this.options.letters;
    this.currentLetter = letters[Math.floor(Math.random() * letters.length)];
    
    // Update UI
    if (this.letterDisplay) {
      this.letterDisplay.textContent = this.currentLetter;
    }
    
    if (this.resultDisplay) {
      this.resultDisplay.textContent = '';
    }
    
    // Reset prediction tracking
    this.lastPrediction = null;
    this.lastPredictionTime = null;
    this.predictionCount = 0;
    
    // Track stats
    this.reloadCount++;
    this.totalQuestions++;
    
    // Check if we need to reload model
    if (this.options.reloadInterval > 0 && this.reloadCount >= this.options.reloadInterval) {
      await this.reloadModel();
      this.reloadCount = 0;
    }
    
    // Fire event
    this.dispatchEvent('newLetter', { letter: this.currentLetter });
    
    return this.currentLetter;
  }
  
  /**
   * Handle a prediction
   * @param {string} prediction - The predicted letter
   */
  handlePrediction(prediction) {
    if (!this.active || !this.currentLetter) return;
    
    if (prediction === this.currentLetter) {
      if (this.lastPrediction === prediction) {
        // Continue timing the same prediction
        const now = new Date().getTime();
        if (now - this.lastPredictionTime >= this.options.duration * 1000) {
          this.predictionCount++;
          
          // Need at least 2 consecutive "stable" predictions
          if (this.predictionCount >= 2) {
            this.correctGuesses++;
            
            if (this.resultDisplay) {
              this.resultDisplay.textContent = `Correct! ${this.correctGuesses}/${this.options.numGuesses}`;
            }
            
            // Fire event
            this.dispatchEvent('correctAnswer', { 
              letter: this.currentLetter,
              correctGuesses: this.correctGuesses 
            });
            
            // Check if we've reached the target
            if (this.correctGuesses >= this.options.numGuesses) {
              this.complete();
              return;
            }
            
            // Move to next letter after a pause
            setTimeout(() => this.nextLetter(), 1000);
          }
        }
      } else {
        // Start timing a new prediction
        this.lastPrediction = prediction;
        this.lastPredictionTime = new Date().getTime();
        this.predictionCount = 1;
        
        // Update UI to show progress
        if (this.resultDisplay) {
          this.resultDisplay.textContent = 'Keep holding...';
        }
      }
    } else {
      // Reset timing for incorrect prediction
      this.lastPrediction = null;
      this.lastPredictionTime = null;
      this.predictionCount = 0;
      
      // Clear progress message
      if (this.resultDisplay) {
        this.resultDisplay.textContent = '';
      }
    }
  }
  
  /**
   * Complete the quiz
   */
  complete() {
    this.active = false;
    
    // Fire completion event
    this.dispatchEvent('quizCompleted', {
      correct: this.correctGuesses,
      skipped: this.skippedQuestions,
      total: this.totalQuestions
    });
  }
  
  /**
   * Skip the current letter
   */
  skipLetter() {
    if (!this.active) return;
    
    this.skippedQuestions++;
    
    // Fire event
    this.dispatchEvent('letterSkipped', { 
      letter: this.currentLetter,
      skipped: this.skippedQuestions 
    });
    
    this.nextLetter();
  }
  
  /**
   * Show the answer for the current letter
   * @returns {Promise<boolean>} - Whether the answer was shown
   */
  async showAnswer() {
    if (!this.currentLetter || !