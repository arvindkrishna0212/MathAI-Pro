import React, { useState, useEffect } from 'react';
import MathSolver from './components/MathSolver';
import LoadingScreen from './components/LoadingScreen';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading time
    setTimeout(() => setLoading(false), 2000);
  }, []);

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <div className="App">
      <div className="gradient-bg">
        <div className="gradients-container">
          <div className="g1"></div>
          <div className="g2"></div>
          <div className="g3"></div>
          <div className="g4"></div>
          <div className="g5"></div>
          <div className="interactive"></div>
        </div>
      </div>
      
      <header className="App-header">
        <div className="header-content">
          <div className="logo-container">
            <div className="logo-icon">∑</div>
            <h1>MathAI Pro</h1>
          </div>
          <p className="subtitle">Advanced AI-powered mathematics solver with intelligent routing</p>
          <div className="header-features">
            <span className="feature-tag">Step-by-Step Solutions</span>
            <span className="feature-tag">Intelligent Routing</span>
            <span className="feature-tag">Learning System</span>
          </div>
        </div>
      </header>
      
      <main className="main-container">
        <MathSolver />
      </main>
      
      <footer className="App-footer">
      </footer>
    </div>
  );
}

export default App;