# 🤖 AI-Driven Customer Support System

Modern support systems often struggle with slow response times, manual misrouting, and inconsistent resolutions. This project solves those challenges using a multi-agent AI pipeline integrated with Gemini LLM, NLP, and ML.

---

## 🚀 Features

- 🧠 Summarizes entire conversations  
- 🏷️ Categorizes issues and extracts key entities  
- 🎯 Assigns accurate priorities and departments  
- 🧾 Recommends resolutions using Retrieval-Augmented Generation (RAG)  
- ⏱️ Predicts resolution time  
- ✅ Suggests ticket status and follow-ups  
- 📊 Gradio-based dashboard + CSV export  

---

## 🛠️ Technologies Used

🖥️ **Programming Language & Environment**  
- **Python**: Core development language  
- **Jupyter Notebook**: For prototyping and experimentation  

📦 **Machine Learning & NLP Libraries**  
- **spaCy**: NLP pipelines and named entity recognition  
- **SentenceTransformers**: Semantic similarity and vector embeddings  
- **FAISS**: Fast Approximate Nearest Neighbour Search for routing  
- **LightGBM**: Efficient gradient boosting for classification/regression  
- **Scikit-learn**: Model building, training, and evaluation  
- **TF-IDF Vectorizer**: Textual feature extraction  

🤖 **AI/LLM Integration**  
- **Gemini API**: Used for advanced summarization and context understanding  

🧪 **Utilities & Support**  
- **Pandas, NumPy**: Data manipulation  
- **Joblib, Pickle**: Model and pipeline persistence  
- **Dotenv**: Environment variable management  
- **Logging**: Debugging and tracking workflow execution  

🖼️ **Interface & Visualization**  
- **Gradio**: Interactive UI for customer support dashboard  

---

## 📂 Project Structure

```
📦 ai-support-assistant
├── README.md                   # Project documentation
├── app.py                      # Core logic with Gradio interface and agents
├── requirements.txt            # Dependency list
├── historical_data_cache_v3/   # Mock KB articles and past ticket data
└── .env                        # API keys and secrets (excluded from Git)
```

---

## 🔧 Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 3. Run the app

```bash
python app.py
```

---

## 🧠 Agents Overview

| Agent               | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| **Summarizer**       | Summarizes the full customer conversation                        |
| **Entity Extractor** | Extracts OS, software, version, and error information            |
| **Category Assigner**| Classifies the core issue into categories (e.g. Billing, Access) |
| **Sentiment Analyzer**| Evaluates the overall tone (frustrated, confused, etc.)        |
| **Priority Assigner**| Labels the issue severity (Low to Critical)                      |
| **Action Extractor** | Identifies steps discussed or expected                           |
| **RAG Resolver**     | Suggests solution using past KB/tickets and LLM reasoning        |
| **Task Router**      | Assigns to appropriate department/team                            |
| **Time Estimator**   | Predicts resolution or response time                              |
| **Status Suggester** | Suggests final ticket status                                      |
| **Follow-up Generator**| Lists pending or next follow-up actions                        |

---

## 📊 Dashboard Features

- 🔍 Filter by **Status**, **Department**, and **Priority**  
- 🆔 Search any ticket by ID and view complete agent analysis  
- 📥 Export all tickets and metadata to `.csv`  
- 🧠 View both **raw** and **processed** AI agent outputs  
- 🖥️ Fully interactive UI via **Gradio** interface  

---

## 📁 Datasets Used

- 🧪 **Synthetic Ticket Data**: Sample customer-agent transcripts  
- 📚 **Knowledge Base Entries**: Pre-written solutions and FAQs  
- 🌐 **Public NLP Datasets**: Used for embedding generation and testing  

---

## 📈 Model Validation

### 🔍 Classification Tasks (e.g., category, sentiment, priority)
- **Metrics**: Accuracy, Precision, Recall, F1-Score  
- **Validation**: `train_test_split`, `KFold Cross-validation`  
- **Libraries**: `Scikit-learn`, `LightGBM`

### ⏱️ Regression Tasks (e.g., time estimation)
- **Metric**: Mean Absolute Error (MAE)  
- **Tools**: Scikit-learn Regression models  
- **Baseline Comparisons**: Mean baseline and business rules

---

## 💼 Use Cases

- 💬 Automating Tier 1 and Tier 2 customer support queries  
- 🧾 Processing billing issues and routing to finance teams  
- 🧑‍💻 Internal IT helpdesk automation for large enterprises  
- 📊 Real-time dashboards for CX/Operations teams  

---

## 🤝 Team

Built during **Hack the Future: A GenAI Sprint Powered by Data**

- 👩 Kritika Bassi
- 👩‍💼 Shreya Singh
  
---

## 📜 License

This project is released under the **MIT License**.  
It is intended for academic and demonstration use only.  
For commercial licensing, please contact the maintainers.

---

## 🙋‍♀️ Contribution Guidelines

- Fork the repo  
- Create a feature branch (`git checkout -b feature-name`)  
- Commit your changes  
- Open a pull request with a proper description
