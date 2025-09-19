import React, { useState } from 'react';
import axios from 'axios';
import { Star, ThumbsUp, MessageSquare, Send, CheckCircle } from 'lucide-react';
import './Feedback.css';

const API_BASE_URL = 'http://localhost:8000';

function Feedback({ solutionId }) {
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [comment, setComment] = useState('');
  const [improvements, setImprovements] = useState([]);
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  const improvementOptions = [
    'More detailed steps',
    'Better explanations',
    'Additional examples',
    'Visual aids or diagrams',
    'Simpler language',
    'More practice problems',
    'Real-world applications',
    'Common mistakes to avoid'
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (rating === 0) return;

    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/feedback`, {
        solution_id: solutionId,
        rating: rating,
        comment: comment,
        improvements: improvements
      });
      setSubmitted(true);
    } catch (err) {
      alert('Error submitting feedback. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div className="feedback-success">
        <CheckCircle className="success-icon" />
        <h4>Thank you for your feedback!</h4>
        <p>Your input helps us improve our solutions.</p>
      </div>
    );
  }

  return (
    <div className="feedback-container">
      <div className="feedback-header">
        <ThumbsUp className="icon" />
        <h4>How helpful was this solution?</h4>
      </div>

      <form onSubmit={handleSubmit} className="feedback-form">
        <div className="rating-section">
          <label>Rate this solution:</label>
          <div className="star-rating">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                type="button"
                key={star}
                className={`star ${star <= (hoverRating || rating) ? 'filled' : ''}`}
                onClick={() => setRating(star)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(0)}
              >
                <Star className="star-icon" />
              </button>
            ))}
          </div>
          <div className="rating-labels">
            <span>Not helpful</span>
            <span>Very helpful</span>
          </div>
        </div>

        <div className="comment-section">
          <label htmlFor="comment">
            <MessageSquare className="icon" />
            Additional comments (optional):
          </label>
          <textarea
            id="comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows="3"
            placeholder="Tell us more about your experience with this solution..."
          />
        </div>

        <div className="improvements-section">
          <label>What could be improved?</label>
          <div className="improvements-grid">
            {improvementOptions.map((option) => (
              <label key={option} className="improvement-checkbox">
                <input
                  type="checkbox"
                  checked={improvements.includes(option)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setImprovements([...improvements, option]);
                    } else {
                      setImprovements(improvements.filter(i => i !== option));
                    }
                  }}
                />
                <span className="checkmark"></span>
                {option}
              </label>
            ))}
          </div>
        </div>

        <button 
          type="submit" 
          disabled={loading || rating === 0}
          className="submit-feedback-btn"
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              Submitting...
            </>
          ) : (
            <>
              <Send className="icon" />
              Submit Feedback
            </>
          )}
        </button>
      </form>
    </div>
  );
}

export default Feedback;