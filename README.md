# MathAI Pro: AI-Enabled Math Tutor with Intelligent Routing

MathAI Pro is an advanced AI-powered mathematics solver that leverages an Agentic-RAG (Retrieval Augmented Generation) architecture. The system intelligently routes mathematical questions to the most appropriate source—whether it's a pre-built knowledge base, real-time web search, or direct LLM generation—providing step-by-step solutions with human-in-the-loop feedback for continuous improvement.

## 🚀 Features

- **Intelligent Question Routing**: Automatically determines the best approach for each question (Knowledge Base, Web Search, or LLM Generation)
- **Vector Database Integration**: Uses Qdrant for efficient storage and retrieval of mathematical solutions
- **Real-time Web Search**: Integrates with Tavily API for up-to-date information when needed
- **AI Safety Guardrails**: Input and output validation to ensure safe and relevant interactions
- **Human-in-the-Loop Learning**: Collects user feedback to improve future responses with DSPy integration
- **Step-by-Step Explanations**: Provides detailed, easy-to-follow mathematical solutions
- **Full-Stack Application**: FastAPI backend with React frontend for seamless user experience
- **Docker Support**: Easy deployment with containerized services
- **Feedback-Based Learning**: Uses DSPy framework for programmatic improvement of LLM responses

## 🏗️ Architecture

The system follows a sophisticated Agentic-RAG architecture:

1. **User Input**: Mathematical questions submitted via the React frontend
2. **Input Guardrail**: Safety and relevance checks on incoming queries
3. **Router Agent**: Intelligent decision-making to route to Knowledge Base, Web Search, or LLM
4. **Knowledge Base Service**: Retrieval from Qdrant vector database for pre-solved problems
5. **Web Search Service**: Real-time search using Tavily API with MCP fallback
6. **Math Agent (LLM)**: Groq-powered solution generation with DSPy feedback integration
7. **Output Guardrail**: Validation of generated solutions for safety and accuracy
8. **Feedback Collection**: User feedback mechanism for continuous learning
9. **Frontend Display**: Interactive presentation of solutions and feedback prompts

## 📁 Project Structure

```
math-routing-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── math_agent.py       # Core math solving agent with DSPy
│   │   │   ├── router.py           # Intelligent routing logic
│   │   │   └── guardrails.py       # Safety and validation
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py          # Pydantic data models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_base.py   # Qdrant integration
│   │   │   ├── web_search.py       # Tavily API + MCP integration
│   │   │   └── feedback.py         # Feedback collection service
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── math_parser.py      # Mathematical expression parsing
│   │   └── config.py               # Application configuration
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Backend containerization
│   └── .env                        # Environment variables (create this)
├── frontend/
│   ├── src/
│   │   ├── App.js                  # Main React application
│   │   ├── components/
│   │   │   ├── MathSolver.js       # Main solver interface
│   │   │   ├── Feedback.js         # Feedback collection component
│   │   │   ├── StepDisplay.js      # Step-by-step solution display
│   │   │   └── LoadingScreen.js    # Loading animation
│   │   └── services/
│   │       └── api.js              # API communication layer
│   ├── package.json                # Node.js dependencies
│   ├── Dockerfile                  # Frontend containerization
│   └── public/
│       └── index.html              # HTML template
├── data/
│   ├── math_dataset.json           # Sample mathematical problems
│   ├── feedback_data.json          # Collected user feedback
│   ├── kb_stats.json               # Knowledge base statistics
│   └── routing_history.json        # Routing decision history
├── docker-compose.yml              # Multi-service orchestration
├── start_server.py                # Direct server startup script
├── add_gsm8k_dataset.py            # Knowledge base population script
└── README.md                       # This file
```

## 🛠️ Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for manual setup)
- Node.js 18+ (for manual setup)
- Groq API Key (for LLM capabilities)
- Tavily API Key (for web search functionality)
- Optional: MCP API credentials for enhanced search

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository_url>
cd math-routing-agent
```

### 2. Environment Setup

Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MCP_API_KEY=your_mcp_api_key_here  # Optional
MCP_ENDPOINT=your_mcp_endpoint_here  # Optional
```

### 3. Launch with Docker (Recommended)

```bash
docker-compose up --build
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Qdrant Dashboard: http://localhost:6333/dashboard

### 4. Manual Setup (Alternative)

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend
npm install
npm start
```

### 5. Populate Knowledge Base (Optional)

To populate the knowledge base with sample problems:

```bash
cd backend
python ../add_gsm8k_dataset.py
```

This will add 100 random samples from the GSM8K dataset to your Qdrant knowledge base.

## 📖 Usage

1. Open your browser to `http://localhost:3000`
2. Enter a mathematical question in the input field
3. Click "Solve" to receive a step-by-step solution
4. Provide feedback to help improve the system

### Example Queries

**Knowledge Base Examples:**
- "What is the area of a circle with radius 5?"
- "Solve for x: 2x + 3 = 7"

**Web Search + LLM Examples:**
- "Explain the concept of eigenvalues and eigenvectors"
- "What is the proof of the Pythagorean theorem?"
- "How to calculate the definite integral of sin(x) from 0 to π?"

## 🔧 API Endpoints

The backend provides the following REST API endpoints:

- `POST /solve` - Solve a mathematical question
- `POST /feedback` - Submit user feedback
- `GET /feedback/stats` - Get feedback statistics
- `GET /search` - Perform web search
- `GET /health` - Health check
- `GET /guardrail/check` - Test input guardrails
- `GET /kb/status` - Knowledge base status

### API Usage Examples

#### Solve a Problem
```bash
curl -X POST "http://localhost:8000/solve" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the derivative of x² + 3x + 5?"}'
```

#### Submit Feedback
```bash
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "solution_id": "sol_123",
       "rating": 5,
       "comment": "Very helpful solution",
       "improvements": ["More examples"]
     }'
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
python -m pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | Yes |
| `TAVILY_API_KEY` | Tavily web search API key | Yes |
| `MCP_API_KEY` | Model Context Protocol API key | No |
| `MCP_ENDPOINT` | MCP endpoint URL | No |
| `QDRANT_URL` | Qdrant database URL | No (defaults to localhost) |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | No (defaults to "math_knowledge_base") |
| `LLM_MODEL` | LLM model to use | No (defaults to "openai/gpt-oss-120b") |

### Advanced Configuration

The system includes several configurable parameters in `backend/app/config.py`:

- **Guardrails**: Blocked keywords, allowed domains, feedback thresholds
- **Feedback System**: Human-in-the-loop settings, learning rates
- **Model Settings**: Embedding models, vector dimensions
- **Routing**: Similarity thresholds, complexity assessment

## 🏃‍♂️ Direct Server Startup

For development or direct execution without Docker:

```bash
python start_server.py
```

This script automatically:
- Sets up the Python path
- Loads environment variables from `backend/.env`
- Starts the uvicorn server on port 8000

## 📊 Knowledge Base Management

### Populating the Knowledge Base

The `add_gsm8k_dataset.py` script populates your Qdrant database with mathematical problems:

```bash
python add_gsm8k_dataset.py
```

This script:
- Downloads 100 random samples from the GSM8K dataset
- Parses solutions into step-by-step format
- Categorizes problems by mathematical type
- Stores them in the Qdrant vector database

### Knowledge Base Statistics

View KB statistics via the API:

```bash
curl http://localhost:8000/kb/status
```

## 🤖 DSPy Integration

MathAI Pro uses DSPy for programmatic prompt optimization and feedback-based learning:

- **Automatic Improvement**: Solutions are improved based on user feedback patterns
- **Learning from Feedback**: The system learns from common improvement requests
- **Confidence Tracking**: Solution confidence scores are tracked and improved over time
- **Feedback Analysis**: Common patterns in feedback are identified and addressed

## 🔍 Web Search Integration

The system supports multiple search backends:

1. **Tavily API**: Primary web search with high-quality results
2. **MCP (Model Context Protocol)**: Alternative search protocol for enhanced context
3. **Fallback Mechanisms**: Automatic fallback if primary search fails

## 🛡️ Safety & Guardrails

- **Input Validation**: Checks for mathematical relevance and safety
- **Output Filtering**: Ensures generated content is appropriate
- **PII Detection**: Prevents personal information leakage
- **Content Classification**: Maintains focus on mathematical topics

## 🔄 Feedback Loop

The human-in-the-loop feedback system:

1. **Collection**: Users rate solutions and provide comments
2. **Analysis**: Feedback is analyzed for common patterns
3. **Improvement**: Solutions are automatically improved based on feedback
4. **Learning**: The system learns from feedback to provide better responses


**MathAI Pro** - Making mathematics accessible through intelligent AI assistance.
