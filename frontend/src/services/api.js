import axios from 'axios';
    
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const solveQuestion = async (question) => {
    try {
        const response = await api.post('/solve', { question });
        return response.data;
    } catch (error) {
        console.error('Error solving question:', error.response ? error.response.data : error.message);
        throw error;
    }
};

export const submitFeedback = async (feedbackData) => {
    try {
        const response = await api.post('/feedback', feedbackData);
        return response.data;
    } catch (error) {
        console.error('Error submitting feedback:', error.response ? error.response.data : error.message);
        throw error;
    }
};

