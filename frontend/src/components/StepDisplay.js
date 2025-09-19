import React, { useEffect } from 'react';
import { ChevronRight, FunctionSquare, Sigma } from 'lucide-react';
import './StepDisplay.css';

function StepDisplay({ steps }) {
  useEffect(() => {
    if (window.MathJax) {
      window.MathJax.typesetPromise();
    }
  }, [steps]);

  return (
    <div className="steps-container">
      <div className="steps-header">
        <Sigma className="icon" />
        <h4>Step-by-Step Solution</h4>
      </div>
      
      <div className="steps-timeline">
        {steps.map((step, index) => (
          <div key={index} className="step-item">
            <div className="step-marker">
              <span className="step-number">{step.step_number}</span>
            </div>
            
            <div className="step-content">
              <div className="step-explanation">
                {step.explanation}
              </div>
              
              {step.formula && (
                <div className="step-formula-card">
                  <div className="formula-header">
                    <FunctionSquare className="icon" />
                    <span>Formula</span>
                  </div>
                  <div className="formula-content">
                    <span className="math-expression">{step.formula}</span>
                  </div>
                </div>
              )}
              
              {step.result && (
                <div className="step-result-card">
                  <div className="result-header">
                    <ChevronRight className="icon" />
                    <span>Result</span>
                  </div>
                  <div className="result-content">
                    <span className="math-expression">{step.result}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default StepDisplay;