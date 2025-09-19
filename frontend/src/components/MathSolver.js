import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import StepDisplay from './StepDisplay';
import Feedback from './Feedback';
import { Calculator, Send, Sparkles, Brain, Globe, Database } from 'lucide-react';
import './MathSolver.css';

const API_BASE_URL = 'http://localhost:8000';

function MathSolver() {
  const [question, setQuestion] = useState('');
  const [solution, setSolution] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [solutionId, setSolutionId] = useState(null);
  const [history, setHistory] = useState([]);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [question]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setSolution(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/solve`, {
        question: question.trim(),
        context: ''
      });

      // Backend returns solution directly
      setSolution(response.data.solution);
      setSolutionId(response.data.solution.id);

      // Add to history
      setHistory(prev => [{
        question: question.trim(),
        solution: response.data.solution,
        timestamp: new Date().toISOString()
      }, ...prev.slice(0, 4)]);
    } catch (err) {
      setError(err.response?.data?.detail || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const loadFromHistory = (item) => {
    setQuestion(item.question);
    setSolution(item.solution);
    setError(null);
  };

  const clearAll = () => {
    setQuestion('');
    setSolution(null);
    setError(null);
    setSolutionId(null);
  };

  return (
    <div className="math-solver-container">
      <div className="solver-header">
        <h2>
          <Calculator className="icon" />
          Mathematics Problem Solver
        </h2>
        <p>Enter any mathematical problem and get detailed step-by-step solutions</p>
      </div>

      <div className="solver-layout">
        <div className="input-section">
          <form onSubmit={handleSubmit} className="question-form">
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter your mathematical problem here... (e.g., What is the derivative of x² + 3x + 5?)"
                rows="3"
                className="question-input"
              />
              <div className="input-actions">
                <button type="button" onClick={clearAll} className="clear-btn">
                  Clear
                </button>
                <button 
                  type="submit" 
                  disabled={loading || !question.trim()}
                  className="solve-btn"
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      Solving...
                    </>
                  ) : (
                    <>
                      <Send className="icon" />
                      Solve Problem
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>

          {history.length > 0 && (
            <div className="history-section">
              <h3>Recent Problems</h3>
              <div className="history-list">
                {history.map((item, index) => (
                  <div 
                    key={index} 
                    className="history-item"
                    onClick={() => loadFromHistory(item)}
                  >
                    <span className="history-question">{item.question}</span>
                    <span className="history-time">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="solution-section">
          {error && (
            <div className="error-card">
              <div className="error-header">
                <span className="error-icon">⚠️</span>
                <h3>Error</h3>
              </div>
              <p>{error}</p>
              <button onClick={() => setError(null)} className="error-dismiss">
                Dismiss
              </button>
            </div>
          )}

          {solution && (
            <div className="solution-card">
              <div className="solution-header">
                <div className="solution-title">
                  <Brain className="icon" />
                  <h3>Solution</h3>
                </div>
                <div className="solution-meta">
                  <span className="confidence-badge">
                    <Sparkles className="icon" />
                    {(solution.confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>
              </div>

              <div className="question-display">
                <strong>Problem:</strong> {solution.question}
              </div>

              <StepDisplay steps={solution.steps} />

              <div className="final-answer-card">
                <div className="answer-header">
                  <Calculator className="icon" />
                  <h4>Final Answer</h4>
                </div>
                <div className="answer-content">{solution.final_answer}</div>
              </div>

              {solution.explanation && (
                <div className="explanation-card">
                  <h4>Explanation</h4>
                  <p>{solution.explanation}</p>
                </div>
              )}

              <div className="solution-footer">
                <span className="category-tag">{solution.category}</span>
                <button 
                  onClick={() => window.print()} 
                  className="print-btn"
                >
                  Print Solution
                </button>
              </div>

              {solutionId && (
                <Feedback solutionId={solutionId} />
              )}
            </div>
          )}

          {!solution && !error && !loading && (
            <div className="empty-state">
              <div className="empty-icon">🧮</div>
              <h3>Ready to solve!</h3>
              <p>Enter a mathematical problem above to get started</p>
              <div className="example-problems">
                <h4>Try these examples:</h4>
                <div className="examples-grid">
                  <button 
                    onClick={() => setQuestion('What is the derivative of x³ + 2x² - 5x + 3?')}
                    className="example-btn"
                  >
                    Derivative problem
                  </button>
                  <button 
                    onClick={() => setQuestion('Solve: x² - 4x + 4 = 0')}
                    className="example-btn"
                  >
                    Quadratic equation
                  </button>
                  <button 
                    onClick={() => setQuestion('Find the area of a circle with radius 7')}
                    className="example-btn"
                  >
                    Geometry problem
                  </button>
                  <button 
                    onClick={() => setQuestion('What is sin(30°) + cos(60°)?')}
                    className="example-btn"
                  >
                    Trigonometry
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MathSolver;
