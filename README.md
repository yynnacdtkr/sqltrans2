<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=NLP2SQL&fontSize=80&animation=fadeIn&fontAlignY=35)

</div>

<div align="center">

[![Made with Azure OpenAI](https://img.shields.io/badge/Made%20with-Azure%20OpenAI-0078D4?style=for-the-badge&logo=microsoft)](https://azure.microsoft.com/en-us/services/cognitive-services/openai/)
[![Made with Gemini](https://img.shields.io/badge/Made%20with-Gemini-FF6F61?style=for-the-badge&logo=google)](https://developers.generativeai.google/)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Database](https://img.shields.io/badge/Database-PostgreSQL%20%7C%20SQLite-336791?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)

> 🎯 Transform natural language into powerful SQL queries with ease!

</div>

<p align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Right.png" alt="Magnifying Glass" width="25" height="25" /> Turn your words into SQL magic
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Chart%20Increasing.png" alt="Chart" width="25" height="25" /> Visualize your data dynamically
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bookmark%20Tabs.png" alt="Tabs" width="25" height="25" /> Multi-database support
</p>

## ✨ Features

<details>
<summary>🎯 Core Features</summary>

- **Natural Language to SQL** 🗣️ → 📝
  - Convert text queries into SQL commands using **Azure OpenAI** or **Gemini** models
  - Intelligent query interpretation with detailed decision logs
  - Step-by-step reasoning for query generation
- **Multimodal Interaction** 🎥
  - Process both textual and visual inputs for richer query context
  - Combine image data with text to enhance query generation
- **Multi-Database Support** 🗄️
  - SQLite compatibility with file upload
  - PostgreSQL integration with secure connection
  - Dynamic schema exploration
- **Interactive Data Explorer** 🔍
  - Real-time data filtering and exploration
  - Comprehensive query results with summary statistics
  - Advanced table views with sorting and filtering

</details>

<details>
<summary>📊 Visualization & Analytics</summary>

- **Dynamic Visualizations** 📈
  - Multiple chart types (Bar, Line, Scatter, Area, Histogram)
  - Interactive chart configuration
  - AI-powered visualization recommendations
- **Summary Statistics** 📋
  - Detailed numeric analysis
  - Categorical data insights
  - Distribution analysis
  - Statistical measures (mean, median, mode, skewness, kurtosis)

</details>

<details>
<summary>🛡️ Security & Management</summary>

- **Safe SQL Execution** 🔒
  - Strict query validation
  - SQL injection prevention
  - Comprehensive error handling and feedback
- **Query History** 📚
  - Searchable query log
  - Query reusability
  - Multiple export formats (CSV, Excel, JSON)

</details>

## 🚀 Getting Started

```mermaid
graph LR
    A[User Input] --> B[LLM Processing]
    B --> C[SQL Generator]
    C --> D[Database]
    D --> E[Results]
    E --> F[Visualization]
```

### Installation

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/lohitkolluri/NLP2SQLL.git
cd NLP2SQL
```

2️⃣ **Set Up Environment**

```bash
# Create .env file
cat << EOF > .env
### 2a: Azure Configuration
LLM_PROVIDER=AZURE
OPENAI_ENDPOINT="https://[ENDPOINT_NAME].openai.azure.com"
OPENAI_API_VERSION="2024-08-01-preview"
OPENAI_API_KEY="YOUR AZURE OPENAI API KEY"
MODEL_NAME="YOUR AZURE OPENAI MODEL NAME"

### 2b: Gemini Configuration
LLM_PROVIDER=GEMINI
GEMINI_API_KEY="YOUR GEMINI API KEY"
EOF
```

3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Launch the App**

```bash
streamlit run app/NLP2SQL.py
```

## 🎨 Built With

<div align="center">

|                                              Technology                                              |    Purpose     |
| :--------------------------------------------------------------------------------------------------: | :------------: |
|     ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit)     | Web Interface  |
| ![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=flat-square&logo=microsoft) | NLP Processing |
|     ![Gemini](https://img.shields.io/badge/Gemini-FF6F61?style=flat-square&logo=google)               | NLP Processing |
|   ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql)    | PostgreSQL DB  |
|         ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite)          |   SQLite DB    |
|       ![Plotly](https://img.shields.io/badge/Plotly-FF7700?style=flat-square&logo=plotly)               | Visualizations |

</div>

## 🌟 Key Features

```mermaid
mindmap
  root((NLP2SQL))
    Query Processing
      Natural Language Input
      Decision Logging
      Detailed Reasoning
    Visualization
      Interactive Charts
      Summary Statistics
      Data Distribution
    Database
      PostgreSQL
      SQLite
      Schema Analysis
    Security
      Query Validation
      Error Handling
      Safe Execution
```

## 💡 How It Works

1. **Query Input** ➡️ User enters natural language query
2. **Processing** ➡️ Azure OpenAI analyzes and generates SQL with reasoning
3. **Validation** ➡️ Query is validated for safety and correctness
4. **Execution** ➡️ Query runs against selected database
5. **Analysis** ➡️ Results are processed with summary statistics
6. **Visualization** ➡️ Data is presented with AI-recommended charts
7. **Export** ➡️ Results can be downloaded in multiple formats

## 📊 Supported Visualizations

- **Bar Chart**: Comparing categorical data
- **Line Chart**: Time-series and trend analysis
- **Scatter Plot**: Relationship between variables
- **Area Chart**: Cumulative totals and trends
- **Histogram**: Distribution analysis

## 🔒 Security Features

- Strict SQL query validation
- Prevention of harmful SQL operations
- Secure database connections
- Protected sensitive information
- Input sanitization

## 📈 Data Analysis

- Comprehensive summary statistics
- Distribution analysis
- Correlation detection
- Trend identification
- Outlier detection

<div align="center">

</div>
<div align="center">

License © [Lohit Kolluri](LICENSE)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" width="100%"/>

</div>
