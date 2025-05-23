# 📚 Local LLM + Ollama + AI Agent 

A powerful agentic AI application that discovers historical events that happened on today's date using Tavily search API and processes them with local LLMs via Ollama and LangGraph.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

![Screenshot 2025-05-23 at 10 34 24 AM](https://github.com/user-attachments/assets/87fa5c2f-0619-4279-bfa1-1a22a0f0b30b)


## 🌟 Features

- 🔍 **Web Search**: Uses Tavily API to search for historical events
- 🤖 **Local LLM Processing**: Processes results with Ollama (runs locally)
- 📊 **Structured Output**: AI-generated structured summaries
- 🎨 **Beautiful UI**: Streamlit-based interface
- 🏠 **Privacy-First**: Runs entirely on your local machine
- 🔧 **Extensible**: Built with LangGraph for easy customization

## 🏗️ Architecture

```
User Input → Tavily Search → LLM Processing → Structured Response
     ↓              ↓              ↓              ↓
  Streamlit → Historical Events → Ollama → Formatted Output
```
## 🌟 Why Local LLMs with Ollama and LangGraph?

Running local LLMs (Large Language Models) with Ollama and LangGraph offers several compelling advantages and serves as a demonstration of the potential of local LLMs in real-world applications. This project showcases how local LLMs can be effectively utilized for analyzing historical events while maintaining privacy and control.

### 🔒 Privacy
- **Data Security**: All processing happens locally on your machine, ensuring that sensitive data never leaves your environment.
- **No External APIs**: Unlike cloud-based solutions, local LLMs eliminate the risk of data breaches or unauthorized access.

### 💰 Cost-Effectiveness
- **No Subscription Fees**: Avoid recurring costs associated with cloud-based LLM services.
- **One-Time Setup**: Once the model is downloaded, there are no additional charges for usage.

### ⚡ Performance
- **Low Latency**: While local processing eliminates network delays, it may be slower overall compared to cloud-based solutions due to limited local compute power.
- **Offline Capability**: Operate seamlessly without an internet connection, making it ideal for remote or restricted environments.

### 🛠️ Customization
- **Model Flexibility**: Choose from a variety of models like `mistral:7b`, `llama2`, or `phi` to suit your specific needs.
- **Fine-Tuning**: Tailor models to your unique requirements for improved accuracy and relevance.

### 🏠 Independence
- **Self-Reliance**: Gain full control over your AI workflows without relying on third-party services.

By combining Ollama's local LLM capabilities with LangGraph's powerful agent framework, this application delivers a robust, private, and efficient solution for analyzing historical events.


## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai
   ```

2. **Pull a Model**
   ```bash
   ollama pull mistral:7b
   # or try: llama2, codellama, phi, etc.
   ```

3. **Get Tavily API Key**
   - Sign up at [tavily.com](https://tavily.com)
   - Get your API key from the dashboard

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/historical-events-ai-agent.git
   cd historical-events-ai-agent
   ```

2. **Create a Virtual Environment**

   1. **Create the virtual environment**
      ```bash
      python3 -m venv venv
      ```

   2. **Activate the virtual environment**
      ```bash
      source venv/bin/activate
      ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export TAVILY_API_KEY=your_tavily_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎯 Usage

1. **Start the app** and ensure Ollama is running
2. **Click "Discover Today's Historical Events"**
3. **Wait for the AI agent** to search and analyze
4. **Read the structured response** with historical insights

## 🔧 Configuration

### Change the LLM Model

Edit the `OLLAMA_MODEL` variable in `app.py`:

```python
OLLAMA_MODEL = "llama2"  # or "codellama", "phi", etc.
```

### Customize Search Parameters

Modify the search tool configuration:

```python
search_tool = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    max_results=10,  # Increase for more results
    search_depth="advanced"  # or "basic"
)
```

## 📁 Project Structure

```
historical-events-ai-agent/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .env.example       # Environment variables template
└── .gitignore         # Git ignore file
```

## 🔍 How It Works

### 1. Search Phase
- Uses Tavily API to search for historical events
- Constructs search query based on current date
- Retrieves multiple sources with rich content

### 2. Analyze Phase
- The `analyze_events` agent processes search results
- Extracts key insights and identifies significant events

### 3. Structure Phase
- The `structure_response` agent organizes the analysis
- Structures information into an engaging and coherent format

### 4. Output Phase
- Presents formatted response via Streamlit
- Shows both processed analysis and raw search data
- Maintains complete privacy (local processing)

## 🛠️ Advanced Usage

### Running Without UI

```python
from app import create_agent_graph
from langchain_core.messages import HumanMessage

# Create agent
agent = create_agent_graph()

# Run agent
result = agent.invoke({
    "messages": [HumanMessage(content="Find historical events")],
    "current_date": "",
    "search_results": "",
    "final_response": ""
})

print(result["final_response"])
```

### Extending the Agent

Add new nodes to the LangGraph workflow:

```python
def validate_sources(state: AgentState) -> AgentState:
    """Validate the credibility of sources"""
    # Your validation logic here
    return state

workflow.add_node("validate", validate_sources)
workflow.add_edge("search", "validate")
workflow.add_edge("validate", "analyze")
```

## 📝 Environment Variables

The required API keys and model configurations are included in the `.env` file. Ensure you create a `.env` file in the root directory with the following content:

```env
TAVILY_API_KEY=your_tavily_api_key_here
OLLAMA_MODEL=mistral:7b
```

The application will automatically load these variables using `python-dotenv`.

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Add support for specific date queries
- Implement caching for repeated searches
- Add more LLM providers
- Create additional output formats
- Add unit tests

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📋 Requirements

- Python 3.8+
- Ollama installed and running
- Tavily API key
- 4GB+ RAM (for local LLM)

## 🔒 Privacy & Security

- All LLM processing happens locally
- No data sent to external LLM APIs
- Only search queries sent to Tavily
- Your conversations stay on your machine

## 📚 Learn More

- [Ollama Documentation](https://ollama.ai/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Tavily API Documentation](https://tavily.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🐛 Troubleshooting

### Common Issues

1. **"Ollama not found"**
   - Ensure Ollama is installed and running: `ollama serve`

2. **"Model not found"**
   - Pull the model: `ollama pull mistral:7b`

3. **"Tavily API error"**
   - Check your API key is set correctly
   - Verify you have API credits remaining

4. **"Connection refused"**
   - Ensure Ollama is running on default port (11434)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM capabilities
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent framework
- [Tavily](https://tavily.com) for search API
- [Streamlit](https://streamlit.io) for the beautiful UI

---

⭐ **Star this repo** if you found it helpful!

📬 **Questions?** Open an issue or reach out!