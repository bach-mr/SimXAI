
# SIMXAI

This repository provides the codebase and evaluation suite to reproduce the results from the paper:

> **SimXAI: A Simple Conversational XAI Framework**

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Run Interactive Chat (Streamlit)

Launch the conversational interface:

```bash
cd agent
streamlit run chat_app.py
```

---

## 📊 Evaluation Suite

The `evaluation_suite/` directory contains datasets and scripts for **LLM-as-a-judge** evaluation. 

It supports evaluation at multiple level:

* **Single-turn** (individual responses)
* **Multi-turn** (short context)
* **Dialogue-level** (full interactions)

at different aspects:
* Parsing
* Faithfulness
* Contextualization

---

## 📁 Repository Structure (Optional)

```
.
├── agent/                 # Chat interface (Streamlit app)
├── evaluation_suite/      # Evaluation data and scripts
├── requirements.txt
└── README.md
```

---

## 📌 Notes

* Ensure all dependencies are installed before running the app.
* The evaluation suite is modular and can be extended for additional benchmarks or metrics.
