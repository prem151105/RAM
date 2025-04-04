# AgentNet Customer Support System

An AI-driven customer support system using a decentralized multi-agent architecture to enhance efficiency and customer satisfaction through automated processes.

## Problem Statement

In modern enterprises, delivering efficient and high-quality customer support is a critical challenge. As businesses scale, customer queries increase in volume and complexity, requiring rapid and accurate resolution. However, customer support teams often struggle with delays in response times, inconsistent resolutions, and inefficient routing of tasks to other teams.

This system aims to enhance efficiency and customer satisfaction by automatically:
- Summarizing conversations and extracting actionable insights
- Recommending resolutions based on historical data
- Routing tasks effectively to the appropriate teams
- Optimizing resolution time through data-driven estimation

## Features

- **Multi-Agent Architecture**: Specialized agents collaborate to handle customer queries
- **Intelligent Task Routing**: Routes queries to appropriate specialized agents based on query content
- **Historical Data Analysis**: Leverages past support tickets to recommend solutions
- **Confidence Scoring**: Provides confidence scores for recommended solutions
- **Resolution Time Estimation**: Estimates resolution time for better customer expectations
- **Web Interface**: Streamlit-based UI for easy interaction with the system

## System Components

1. **Agent Network**: Coordinates specialized agents to process customer queries
2. **Specialized Agents**: Each agent has specific capabilities for different support areas
3. **Historical Data Processing**: Analyzes past tickets for similar cases and solutions
4. **Ollama Integration**: Uses local LLMs for text processing and generation

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) with at least one model (llama2 recommended)
- Streamlit and other dependencies (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AgentNet-Customer-Support.git
cd AgentNet-Customer-Support
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and set up Ollama:
   - Download Ollama from [ollama.ai](https://ollama.ai/)
   - Install and run the Ollama application
   - Pull the llama2 model (or another model of your choice):
   ```bash
   ollama pull llama2
   ```

## Usage

### Running the Streamlit Application

The Streamlit application provides a web interface to interact with the agent network:

```bash
streamlit run app.py
```

This will launch a web interface at http://localhost:8501 where you can:
- Submit customer support queries
- View agent responses and routing paths
- See confidence scores and estimated resolution times
- Explore historical data

### Using the CLI

For command-line usage, you can use the CLI interface:

```bash
# Process a single query
python run.py cli query "I'm having trouble installing your software"

# Process a batch of queries from a data file
python run.py cli batch --file data/raw/support_queries.csv --limit 5

# Generate test data
python run.py cli generate --count 100
```

### Running the API Server

To run the API server for integration with other systems:

```bash
python run.py server
```

The API will be available at http://localhost:8000 with the following endpoints:
- `POST /support`: Submit a customer support request
- `GET /network/status`: Get current network status

## Project Structure

- `app.py`: Streamlit web application
- `run.py`: Main script for running the server or CLI
- `src/`: Core source code
  - `network/`: Agent network implementation
  - `agent/`: Agent components (router, executor, memory)
  - `utils/`: Utility functions and classes
  - `prompts/`: Prompt templates for LLM
- `data/`: Data files
  - `raw/`: Raw data files including historical tickets and templates
- `tests/`: Test scripts
- `configs/`: Configuration files

## Troubleshooting

### Ollama Connection Issues

If the system can't connect to Ollama:
1. Ensure Ollama is running in the background
2. Verify that the llama2 model (or your chosen model) is installed
3. Check that Ollama is accessible at http://localhost:11434

### Missing Data Files

If you encounter issues with missing data files:
1. Check that the `data/raw/Historical_ticket_data.csv` file exists
2. Ensure template files are present in the `data/raw/` directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the AgentNet framework for decentralized multi-agent systems
- Uses Ollama for LLM integration
- Implements RAG mechanism for improved decision making 