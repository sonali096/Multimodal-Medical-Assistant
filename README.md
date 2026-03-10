# 🏥 Multimodal Medical Assistant

**Capstone Project CS[04]** - Production-Grade Clinical AI System

An AI-powered clinical decision support system that processes multimodal medical data (text + images) using state-of-the-art medical AI models for diagnostic support, triage assessment, and clinical information retrieval.

---

## 🌟 Key Features

### 🤖 Advanced AI Models
- **Llama 2/3** (via Ollama): Natural language understanding and response generation
- **BiomedCLIP/MedCLIP**: Clinical image analysis (X-ray, CT, MRI, pathology)
- **BioBERT/ClinicalBERT**: Medical text encoding and semantic understanding
- **LangChain**: Cross-modal orchestration and query routing

### 📊 Clinical Capabilities
- **Multimodal Query Processing**: Combined text + image analysis
- **HIPAA-Compliant Data Handling**: Automated PHI de-identification
- **Triage Assessment**: Automated patient urgency classification
- **Diagnostic Support**: Evidence-based diagnostic suggestions
- **Medical Knowledge Retrieval**: Vector-based semantic search
- **DICOM Support**: Native processing of medical imaging standards

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│           CLINICAL INTERFACE (Chatbot/API)              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│      LANGCHAIN ORCHESTRATION (Query Routing)            │
└────────┬──────────────────────────────┬─────────────────┘
         │                              │
┌────────▼────────┐          ┌──────────▼─────────┐
│  TEXT PIPELINE  │          │  VISION PIPELINE   │
│  • Llama 2/3    │          │  • BiomedCLIP      │
│  • BioBERT      │          │  • MedCLIP         │
│  • ClinicalBERT │          │  • OpenCV          │
└────────┬────────┘          └──────────┬─────────┘
         │                              │
         └──────────┬───────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│          MULTIMODAL FUSION (Cross-modal embeddings)      │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│    DATA INGESTION (PHI removal + DICOM processing)       │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

**1. Python 3.9 or higher**

**2. Install Ollama** (for Llama 2/3 integration)
```bash
# Visit https://ollama.ai to download and install
# Then pull Llama model:
ollama pull llama2
```

### Installation Steps

**1. Navigate to project directory**
```bash
cd Capstone_Project-CS[04]
```

**2. Create virtual environment**
```bash
python -m venv .venv

# Activate (macOS/Linux):
source .venv/bin/activate

# Activate (Windows):
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Download spaCy model** (for PHI de-identification)
```bash
python -m spacy download en_core_web_sm
```

**5. Verify installation**
```bash
python main.py --mode interactive
```

---

## 💻 Usage Examples

### 1. Interactive Chatbot Demo (Recommended First Run)
```bash
python main.py --mode interactive
```
This runs a complete demo with sample queries showcasing all system capabilities.

### 2. Start API Server
```bash
python main.py --mode api --port 8000
```
Access the interactive API documentation at: http://localhost:8000/docs

### 3. Multimodal Query (Text + Image)
```bash
python main.py --mode query \
  --text "What is the likely diagnosis?" \
  --image ./data/chest_xray.dcm \
  --context "Patient presents with fever, cough, dyspnea"
```

### 4. Text-Only Medical Query
```bash
python main.py --mode query \
  --text "Explain treatment options for community-acquired pneumonia" \
  --context "65-year-old patient with diabetes and hypertension"
```

### 5. Image-Only Analysis
```bash
python main.py --mode query \
  --image ./data/chest_xray.png
```

### 6. Data Ingestion with PHI De-identification
```bash
python main.py --mode ingest \
  --ehr-file ./data/clinical_notes.txt \
  --dicom-dir ./data/scans/
```

---

## 📋 Available Modes

| Mode | Description | Usage |
|------|-------------|-------|
| **interactive** | Run chatbot demo with sample scenarios | `python main.py --mode interactive` |
| **api** | Start FastAPI server | `python main.py --mode api --port 8000` |
| **query** | Single multimodal query | `python main.py --mode query --text "..." --image "..."` |
| **ingest** | Process medical files with PHI removal | `python main.py --mode ingest --ehr-file "..."` |
| **train** | Train custom models | `python main.py --mode train` |
| **evaluate** | Evaluate model performance | `python main.py --mode evaluate` |
| **predict** | Batch predictions | `python main.py --mode predict --text "..." --image "..."` |

---

## 🔒 HIPAA Compliance & Security

### Automated PHI De-identification
The system automatically removes Protected Health Information (PHI):
- **Names**: Replaced with `[PATIENT_X]`
- **Dates**: Replaced with `[DATE]`
- **Phone Numbers**: Replaced with `[PHONE]`
- **Email Addresses**: Replaced with `[EMAIL]`
- **Social Security Numbers**: Replaced with `[SSN]`

### Example
```python
from ingestion import MedicalTextIngestion

processor = MedicalTextIngestion()
text = "Patient John Smith (DOB: 01/15/1980, SSN: 123-45-6789) presents..."

result = processor.process_ehr_text(text)
# Output: "Patient [PATIENT_1] (DOB: [DATE], SSN: [SSN]) presents..."
```

### Security Features
✅ Encryption at rest and in transit  
✅ Complete audit logging  
✅ Role-based access control (RBAC)  
✅ Configurable data retention policies  

---

## 📊 Models & Technologies

### Text Processing
| Model | Purpose | Parameters |
|-------|---------|------------|
| **Llama 2/3** | Natural language generation, clinical reasoning | 7B-70B |
| **BioBERT** | Medical text embeddings | 110M |
| **ClinicalBERT** | Clinical note understanding | 110M |
| **Bio_ClinicalBERT** | Combined biomedical + clinical | 110M |

### Image Processing
| Model | Purpose | Supported Formats |
|-------|---------|-------------------|
| **BiomedCLIP** | Medical image analysis | DICOM, PNG, JPEG |
| **MedCLIP** | Image-text matching | X-ray, CT, MRI |
| **OpenCV** | Image preprocessing | All standard formats |
| **Pydicom** | DICOM parsing | DICOM (.dcm) |

### Orchestration
- **LangChain**: Query routing, prompt templating, chain management
- **FAISS**: Vector similarity search for knowledge retrieval
- **Sentence Transformers**: High-quality embeddings

---

## 📁 Project Structure

```
Capstone_Project-CS[04]/
│
├── main.py                     # Main entry point - start here!
├── config.py                   # Production configuration
├── requirements.txt            # Dependencies
│
├── ingestion.py               # Data ingestion & PHI de-identification
├── cross_modal.py             # Multimodal pipeline (text + vision)
├── clinical_interface.py      # Chatbot & dashboard
│
├── models.py                  # Model architectures
├── train.py                   # Training module
├── inference.py               # Prediction module
├── evaluate.py                # Evaluation & metrics
├── api.py                     # FastAPI endpoints
├── utils.py                   # Utility functions
│
├── data/                      # Data directory
│   ├── raw/                   # Raw medical data
│   ├── processed/             # Processed data
│   └── knowledge_base.faiss   # Vector store
│
└── .venv/                     # Virtual environment
```

---

## 🧪 Testing

### Test Individual Components
```bash
# Test data ingestion
python ingestion.py

# Test cross-modal processing
python cross_modal.py

# Test clinical interface
python clinical_interface.py
```

### Run Full Test Suite
```bash
pytest tests/ -v --cov
```

---

## 🔧 Configuration

Edit `config.py` to customize:

### Model Settings
```python
LLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',  # Ollama endpoint
    'model_name': 'llama2',                # Model to use
    'temperature': 0.7,                    # Response randomness
    'max_tokens': 4096                     # Context window
}
```

### Privacy Settings
```python
PRIVACY_CONFIG = {
    'enable_deidentification': True,       # Auto PHI removal
    'enable_encryption': True,             # Encrypt data
    'enable_audit_logging': True,          # Log all queries
    'data_retention_days': 90             # Retention period
}
```

---

## 🌐 API Endpoints

Once the API server is running (`python main.py --mode api`), access:

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

**Analyze Clinical Text**
```bash
curl -X POST "http://localhost:8000/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient with fever and productive cough",
    "context": "45yo male, smoker"
  }'
```

**Analyze Medical Image**
```bash
curl -X POST "http://localhost:8000/analyze-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "./data/chest_xray.png",
    "image_type": "chest_xray"
  }'
```

**Multimodal Query**
```bash
curl -X POST "http://localhost:8000/multimodal-query" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the diagnosis?",
    "context": "Patient has SOB and chest pain",
    "image_path": "./data/ct_scan.dcm"
  }'
```

---

## 📈 Performance Metrics

The system tracks comprehensive metrics:

### Clinical Metrics
- Sensitivity (Recall)
- Specificity
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)

### ML Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### Operational Metrics
- Query response time
- PHI removal success rate
- DICOM processing success rate

---

## 🛠️ Advanced Usage

### Adding Custom Medical Knowledge

```python
from clinical_interface import MedicalKnowledgeBase

# Initialize knowledge base
kb = MedicalKnowledgeBase()

# Add clinical guidelines
kb.add_documents(
    texts=[
        "Community-acquired pneumonia treatment guidelines...",
        "CURB-65 score: Confusion, Urea >7mmol/L, Resp rate ≥30..."
    ],
    metadatas=[
        {"source": "ATS 2023 Guidelines", "type": "guideline"},
        {"source": "Clinical Reference", "type": "score"}
    ]
)

# Search knowledge base
results = kb.search("pneumonia severity assessment", k=5)
for doc, score in results:
    print(f"Score: {score:.3f} | {doc[:100]}...")
```

### Training Custom Models

```bash
# Train with custom configuration
python main.py --mode train --config custom_config.yaml

# Evaluate trained model
python main.py --mode evaluate --config custom_config.yaml
```

---

## 🐛 Troubleshooting

### Ollama Connection Error
```
⚠️  Llama not available (Ollama not running)
```
**Solution**: Start Ollama service
```bash
ollama serve  # Start Ollama server
ollama pull llama2  # Pull model if not already done
```

### DICOM Processing Error
```
⚠️  pydicom not available
```
**Solution**: Install pydicom
```bash
pip install pydicom
```

### spaCy Model Missing
```
Can't find model 'en_core_web_sm'
```
**Solution**: Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### CUDA/GPU Issues
The system automatically falls back to CPU if GPU is not available. To force CPU:
```python
import torch
torch.device('cpu')  # Already handled in code
```

---

## 📚 Documentation

- **Configuration**: See `config.py` for all configurable parameters
- **API Reference**: Run API mode and visit http://localhost:8000/docs
- **Model Details**: Check docstrings in `cross_modal.py` and `models.py`
- **Data Format**: See examples in `data/` directory

---

## 🎯 Example Workflows

### Workflow 1: Emergency Triage
```bash
# 1. Ingest patient data
python main.py --mode ingest --ehr-file patient_note.txt

# 2. Perform triage assessment
python clinical_interface.py  # Uses triage templates

# 3. Get diagnostic suggestions
python main.py --mode query --text "Triage assessment" --context "..."
```

### Workflow 2: Radiology Consultation
```bash
# 1. Process DICOM image
python main.py --mode query --image chest_ct.dcm

# 2. Add clinical context
python main.py --mode query \
  --image chest_ct.dcm \
  --text "Is this pneumonia or tumor?" \
  --context "Patient: 60yo smoker, chronic cough"
```

### Workflow 3: Knowledge Retrieval
```bash
# Query medical knowledge base
python main.py --mode query \
  --text "What are the Ottawa ankle rules?"
```

---

## 🤝 Contributing

This is an educational capstone project (CS[04]). For academic collaboration inquiries, please contact the project team.

---

## 📜 License

Educational use only - Capstone Project CS[04]

---

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and model hosting
- **Ollama Team**: Local LLM deployment platform
- **Microsoft Research**: BiomedCLIP development
- **Emily Alsentzer et al.**: Bio_ClinicalBERT
- **LangChain**: Orchestration framework
- **OpenAI**: CLIP architecture inspiration

---

## 📞 Support & Contact

For questions or issues:
1. Check this README for common solutions
2. Review code docstrings for detailed API documentation
3. Run interactive demo: `python main.py --mode interactive`
4. Contact the CS[04] capstone project team

---

**🏥 Built with ❤️ for advancing clinical AI and improving patient care**

