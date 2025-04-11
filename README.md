# 🤖 AI-Driven Customer Support System

Modern support systems often struggle with slow response times, manual misrouting, and inconsistent resolutions. This project solves those challenges using a multi-agent AI pipeline integrated with Gemini LLM, NLP, and ML.

## 🚀 Features

- 🧠 Summarizes entire conversations
- 🏷️ Categorizes issues and extracts key entities
- 🎯 Assigns accurate priorities and departments
- 🧾 Recommends resolutions using Retrieval-Augmented Generation (RAG)
- ⏱️ Predicts resolution time
- ✅ Suggests ticket status and follow-ups
- 📊 Gradio-based dashboard + CSV export

---

## 🛠️ Tech Stack

- **Language**: Python
- **UI**: Gradio
- **LLM**: Gemini (Google Generative AI)
- **NLP**: spaCy, SentenceTransformers
- **ML**: LightGBM, Scikit-learn
- **Search & Retrieval**: FAISS, TF-IDF
- **Pipeline Framework**: LangChain
- **Utils**: Pandas, NumPy, dotenv, joblib, pickle

---

## 📂 Project Structure

---

## 📊 Dashboard Features

- 🔍 Filter by **Status**, **Department**, and **Priority**  
- 🆔 Search any ticket by ID and view complete analysis  
- 📥 Export entire ticket list to `.csv`  
- 🧠 View both **raw** and **processed** AI agent outputs  
- 🖥️ Fully interactive and intuitive **Gradio interface**  

---

## 📁 Datasets Used

- 🧪 Synthetic ticket data (mock support conversations)  
- 📚 Knowledge base entries (e.g., past solutions, FAQs)  
- 🌐 Public NLP datasets for model tuning and embeddings  

---

## 📈 Model Validation

### 🧪 Classification Tasks
- **Metrics**: Accuracy, F1-Score  
- **Tools**: Scikit-learn, LightGBM  

### ⏱️ Regression Tasks (e.g., time estimation)
- **Metric**: Mean Absolute Error (MAE)  
- **Validation Techniques**: `KFold`, `train_test_split` cross-validation  

---

## 💼 Use Cases

- Enterprise customer service automation  
- SaaS product support ticketing  
- HR or IT helpdesk platforms  
- Scalable LLM-powered support routing  

---

## 🤝 Team

Built during **Hack the Future: A GenAI Sprint Powered by Data**

- Shreya Singh 
- Kritika Bassi   
