import React from 'react';
import { Sigma, FunctionSquare, Calculator, Infinity } from 'lucide-react';
import './LoadingScreen.css';

function LoadingScreen() {
  const mathSymbols = [
    { icon: <Sigma />, delay: 0 },
    { icon: <FunctionSquare />, delay: 0.2 },
    { icon: <Calculator />, delay: 0.4 },
    { icon: <Infinity />, delay: 0.6 }
  ];

  return (
    <div className="loading-screen">
      <div className="loading-container">
        <div className="math-symbols">
          {mathSymbols.map((symbol, index) => (
            <div
              key={index}
              className="symbol"
              style={{ animationDelay: `${symbol.delay}s` }}
            >
              {symbol.icon}
            </div>
          ))}
        </div>
        
        <div className="loading-text">
          <h1>MathAI Pro</h1>
          <p>Preparing your advanced mathematics solver...</p>
        </div>
        
        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
        
        <div className="loading-features">
          <span>Intelligent Routing</span>
          <span>Step-by-Step Solutions</span>
          <span>Learning System</span>
        </div>
      </div>
    </div>
  );
}

export default LoadingScreen;