"""
Configuration for Multimodal Medical Assistant
Production-grade settings for all components
"""

# ==================== MODEL CONFIGURATIONS ====================

# Llama Configuration (via Ollama)
LLAMA_CONFIG = {
    'model_name': 'llama2',  # or 'llama3' when available
    'base_url': 'http://localhost:11434',  # Ollama default endpoint
    'temperature': 0.7,
    'max_tokens': 2048,
    'context_window': 4096
}

# BiomedCLIP / MedCLIP Configuration
MEDCLIP_CONFIG = {
    'model_name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    'image_size': 224,
    'batch_size': 8,
    'device': 'cpu'  # Will auto-detect CUDA/MPS
}

# BioBERT / ClinicalBERT Configuration
TEXT_ENCODER_CONFIG = {
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
    'max_length': 512,
    'batch_size': 16,
    'embedding_dim': 768
}

# Sentence Transformers Configuration
SENTENCE_TRANSFORMER_CONFIG = {
    'model_name': 'pritamdeka/S-PubMedBert-MS-MARCO',  # Medical domain
    'max_seq_length': 512,
    'device': 'cpu'
}

# ==================== LANGCHAIN CONFIGURATIONS ====================

LANGCHAIN_CONFIG = {
    'chain_type': 'stuff',  # or 'map_reduce', 'refine'
    'verbose': True,
    'temperature': 0.3,
    'max_iterations': 3
}

# ==================== DATA PROCESSING CONFIGURATIONS ====================

# Medical Image Processing
IMAGE_PROCESSING_CONFIG = {
    'supported_formats': ['dcm', 'dicom', 'png', 'jpg', 'jpeg'],
    'dicom_window_center': 40,
    'dicom_window_width': 400,
    'normalize': True,
    'resize': (224, 224)
}

# Text Processing & De-identification
TEXT_PROCESSING_CONFIG = {
    'deidentify': True,
    'phi_types': ['PERSON', 'DATE', 'PHONE', 'EMAIL', 'SSN', 'MRN'],
    'spacy_model': 'en_core_web_sm',
    'medical_ner_model': 'en_ner_bc5cdr_md'
}

# ==================== SECURITY & PRIVACY ====================

PRIVACY_CONFIG = {
    'enable_deidentification': True,
    'hipaa_compliant': True,
    'encrypt_storage': True,
    'audit_logging': True,
    'data_retention_days': 90
}

# ==================== STORAGE CONFIGURATIONS ====================

STORAGE_CONFIG = {
    'data_dir': './data',
    'upload_dir': './data/uploads',
    'processed_dir': './data/processed',
    'embeddings_dir': './data/embeddings',
    'cache_dir': './data/cache',
    'max_file_size_mb': 100
}

# ==================== API CONFIGURATIONS ====================

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 4,
    'timeout': 300,
    'cors_origins': ['http://localhost:3000'],
    'rate_limit': '100/minute'
}

# ==================== VECTOR STORE CONFIGURATIONS ====================

VECTOR_STORE_CONFIG = {
    'type': 'faiss',  # or 'chroma', 'pinecone'
    'dimension': 768,
    'metric': 'cosine',
    'index_type': 'IVFFlat',
    'nlist': 100
}

# ==================== CLINICAL DECISION SUPPORT ====================

CLINICAL_CONFIG = {
    'triage_levels': ['Critical', 'Urgent', 'Semi-Urgent', 'Non-Urgent', 'Routine'],
    'confidence_threshold': 0.75,
    'max_differential_diagnoses': 5,
    'enable_explainability': True
}

# ==================== LOGGING CONFIGURATIONS ====================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './logs/medical_assistant.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# ==================== PROMPT TEMPLATES ====================

PROMPT_TEMPLATES = {
    'text_only': """You are a medical AI assistant. Based on the following clinical information, provide a comprehensive analysis.

Clinical Information:
{context}

Question: {question}

Provide a detailed medical analysis including:
1. Key findings
2. Differential diagnoses
3. Recommended next steps
4. Relevant clinical guidelines

Analysis:""",

    'image_only': """Analyze this medical image and provide a detailed radiological report.

Image Type: {image_type}
Clinical Context: {context}

Provide:
1. Image quality assessment
2. Key findings and abnormalities
3. Anatomical structures identified
4. Diagnostic impressions
5. Recommendations

Report:""",

    'multimodal': """You are an expert medical AI assistant analyzing both clinical text and medical images.

Clinical Text:
{text_context}

Image Analysis:
{image_context}

Question: {question}

Provide an integrated multimodal analysis:
1. Correlation between text findings and image findings
2. Comprehensive diagnostic impression
3. Risk assessment
4. Clinical recommendations
5. Follow-up suggestions

Integrated Analysis:""",

    'triage': """Based on the provided clinical information, perform medical triage assessment.

Patient Information:
{patient_info}

Symptoms: {symptoms}
Vital Signs: {vitals}
Medical History: {history}

Provide:
1. Triage Level (Critical/Urgent/Semi-Urgent/Non-Urgent/Routine)
2. Severity Assessment
3. Time-sensitive concerns
4. Recommended care pathway

Triage Assessment:"""
}

# ==================== MEDICAL ENTITIES ====================

MEDICAL_ENTITIES = {
    'conditions': ['disease', 'syndrome', 'disorder', 'condition'],
    'symptoms': ['symptom', 'sign', 'complaint'],
    'medications': ['drug', 'medication', 'prescription'],
    'procedures': ['procedure', 'surgery', 'intervention'],
    'anatomy': ['organ', 'tissue', 'anatomical_structure'],
    'lab_values': ['lab_test', 'biomarker', 'vital_sign']
}

# ==================== DEFAULT VALUES ====================

DEFAULT_CONFIG = {
    'mode': 'api',
    'debug': False,
    'use_gpu': True,
    'cache_enabled': True,
    'async_processing': True,
    'max_concurrent_requests': 10
}
