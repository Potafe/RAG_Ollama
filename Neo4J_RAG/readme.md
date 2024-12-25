# Neo4j RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot implementation using Neo4j, LangChain, and Ollama.

## Prerequisites

- Docker and Docker Compose
- At least 8GB of RAM available for Docker
- Windows, Linux, or macOS with Docker installed

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up environment variables (optional):
```bash
# Create a .env file with your configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LLM=llama2:13b  # or any other Ollama model
EMBEDDING_MODEL=sentence_transformer
```

3. Start the services:
```bash
docker-compose up
```

4. Access the applications:
- Chatbot UI: http://localhost:8503
- Neo4j Browser: http://localhost:7474

## Architecture

The system consists of several Docker containers:

- **Bot**: A Streamlit application that serves as the chat interface
- **Database**: Neo4j graph database for storing and retrieving knowledge
- **LLM**: Ollama service for running the language model (Linux only)
- **LLM-GPU**: GPU-enabled version of the Ollama service (Linux with NVIDIA GPU only)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEO4J_USERNAME | Neo4j database username | neo4j |
| NEO4J_PASSWORD | Neo4j database password | password |
| NEO4J_URI | Neo4j connection URI | neo4j://database:7687 |
| OLLAMA_BASE_URL | URL for Ollama service | http://host.docker.internal:11434 |
| LLM | Ollama model to use | llama3.2:3b |
| EMBEDDING_MODEL | Model for text embeddings | sentence_transformer |

### Profiles

The docker-compose configuration includes different profiles:

- `linux`: For running on Linux with CPU
- `linux-gpu`: For running on Linux with NVIDIA GPU support

## Development

To make changes to the bot:

1. Modify the Python files:
   - `bot.py`: Main Streamlit application
   - `utils.py`: Utility functions
   - `chains.py`: LangChain implementations

2. The application will automatically rebuild when changes are detected thanks to the `x-develop` configuration.

3. To visit the database go to site: https://localhost:7474

## Data Persistence

- Neo4j data is stored in a Docker volume named `neo4j_data`
- To reset the database, you can remove the volume:
  ```bash
  docker-compose down -v
  ```

## Troubleshooting

### Common Issues

1. **Neo4j fails to start**
   - Ensure no other services are using ports 7474 or 7687
   - Check if you have sufficient disk space
   - Verify Docker has enough memory allocated

2. **Cannot connect to Ollama**
   - Make sure Ollama is running and accessible
   - Check the OLLAMA_BASE_URL environment variable
   - Verify the model specified in LLM environment variable is available

3. **Bot container fails to start**
   - Check the logs using `docker-compose logs bot`
   - Verify all required Python packages are in requirements.txt
   - Ensure Neo4j is healthy before the bot starts

