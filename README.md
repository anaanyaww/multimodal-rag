# Enterprise Multimodal RAG System
## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/5TbEv2F-7x4/maxresdefault.jpg)](https://youtu.be/5TbEv2F-7x4?si=Yc2XHCB56a1TBbfJ)

> Click the image above to watch the full walkthrough on YouTube.


A multimodal Retrieval-Augmented Generation system that processes documents and audio files with fine-tuning capabilities.

##  Features

- **Multimodal Processing**: PDFs (text/tables/images) and audio files
- **Advanced RAG**: BGE embeddings with ChromaDB vector storage
- **Fine-tuning**: Parameter-efficient LoRA implementation
- **Production Ready**: Flask deployment with REST APIs
- **Enterprise Grade**: Error recovery, caching, and monitoring

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Ollama with LLaVA model
- 8GB+ RAM

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anaanyaww/multimodal-rag.git
cd multimodal-rag

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llava:7b
