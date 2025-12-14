# Farmer Friend – Multimodal AI-Based Agricultural Advisory System

Farmer Friend is a locally deployable, multimodal agricultural advisory system designed to assist farmers with crop-related decision making using text, voice, and image inputs. The system combines a large language model with computer vision, speech recognition, weather intelligence, and structured agricultural datasets to provide reliable, context-aware recommendations.

The primary goal of this project is to reduce farmers’ dependence on scarce agricultural experts by delivering accessible, multilingual, and real-time advisory support that can run on consumer-grade hardware without cloud dependency.

## Key Capabilities
- Multilingual conversational support (English, Hindi, Kannada) 
- Voice-based interaction using on-device speech recognition
- Leaf image–based plant health and disease analysis
- Retrieval-Augmented Generation (RAG) for factual responses
- Real-time weather-aware recommendations
- Local execution using open-source models (no paid APIs)

## System Overview
The system accepts three types of farmer inputs:
* Text queries for crop, fertilizer, subsidy, market, and disease-related questions
* Voice input captured via the browser microphone and transcribed using Whisper
* Leaf images uploaded for plant stress and disease analysis
These inputs are processed through dedicated pipelines and combined with external knowledge sources and weather data before being passed to a locally hosted LLM (Llama 3.1 via Ollama) for reasoning and response generation.

## Project Structure
```
CHATBOT/
│
├── data/
│   ├── plant_disease/
│   │   └── test/
│   │       ├── Alstonia Scholaris/
│   │       ├── Arjun/
│   │       ├── Bael/
│   │       ├── Basil/
│   │       ├── Chinar/
│   │       ├── Gauva/
│   │       ├── Jamun/
│   │       ├── Jatropha/
│   │       ├── Lemon/
│   │       ├── Mango/
│   │       ├── Pomegranate/
│   │       └── Pongamia Pinnata/
│   │
│   ├── data_core.csv
│   ├── Indian Rainfall Dataset District-wise Daily.csv
│   ├── mandi_data_2000_rows.csv
│   ├── subsidy.txt        
│
├── vectors/               ← auto-generated embeddings 
├── disease_model.h5
├── app.py                  ← main Streamlit app
├── train_disease_model.py
├── build_plant_vectors.py
├── ragbuilt.py             ← builds RAG index from datasets
├── requirements.txt
├── README.md 
```
## Datasets Used
Due to large file sizes, the datasets used in this project are not included in the repository. Please download them manually using the links below and place them inside the data/ directory.
- Crop and Soil Dataset
Used for crop suitability, soil properties, and fertilizer recommendations.
https://www.kaggle.com/datasets/shankarpriya2913/crop-and-soil-dataset
- Mandi Price Dataset
Used for market trend analysis and selling-time recommendations.
https://www.kaggle.com/datasets/skm1234556/mandi-data
- Indian Rainfall Dataset
Used for irrigation guidance, drought risk, and weather-aware advice.
https://www.kaggle.com/datasets/ankitgaikar1995/imd-rainfall-dataset-2022
- Plant Disease Image Dataset
Used to train the MobileNetV2-based leaf disease classification model.
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Government Subsidy Dataset (Custom)
File: subsidy.txt
Manually curated from official central and state government portals. Includes PM-KISAN, PMFBY, fertilizer subsidies, electricity waivers, and state-specific schemes. Covers updates for 2024–2025. This file is included in the repository.

## Models and Technologies
* Large Language Model: Llama 3.1 (8B) via Ollama
* Speech Recognition: Whisper Tiny (CPU-friendly)
* Vision Model: MobileNetV2 (fine-tuned for leaf analysis)
* Weather Data: Open-Meteo API
* Frontend: Streamlit
* Execution Mode: Local, CPU-only 

## Setup Instructions
- Install Dependencies
```pip install streamlit langchain langchain_ollama whisper gTTS pillow numpy pandas requests librosa soundfile tensorflow ```
- Install Ollama and Pull Model
``` ollama pull llama3.1:8b ```
- Start Ollama Server
```ollama serve```
(If port 11434 is unavailable, configure a custom port using the OLLAMA_HOST environment variable.)
- Run the Application
```streamlit run app.py```

## Performance Summary

- Leaf disease classification accuracy: ~92%
- End-to-end response latency: < 4 seconds (CPU-only)
- Supports code-mixed and regional speech input
- Operates offline except for weather data retrieval

## Future Scope

- Vision Transformers and segmentation-based plant analysis
- Domain-specific LLM fine-tuning using ICAR/KVK datasets
- Integration with IoT sensors (soil moisture, pH, weather stations)
- Android application and WhatsApp chatbot deployment
- Fully offline mobile inference using quantized models
