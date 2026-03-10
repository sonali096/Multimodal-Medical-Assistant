"""
Cross-Modal Processing Pipeline
Integrates text (Llama, BioBERT) and vision (BiomedCLIP) modalities

Components:
1. Text Processing: Llama 2/3, BioBERT, ClinicalBERT
2. Image Processing: BiomedCLIP, MedCLIP
3. Multimodal Fusion: Cross-modal embeddings and reasoning
4. LangChain Orchestration: Query routing and response generation
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Deep learning
import torch
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# LangChain
try:
    from langchain.llms import Ollama
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: pip install langchain")

# Image processing
from PIL import Image
import cv2

# Configuration
from config import (
    LLAMA_CONFIG, MEDCLIP_CONFIG, TEXT_ENCODER_CONFIG,
    SENTENCE_TRANSFORMER_CONFIG, LANGCHAIN_CONFIG, PROMPT_TEMPLATES
)


class LlamaTextProcessor:
    """
    Llama 2/3 processor for natural language understanding
    Connects to Ollama API for inference
    """
    
    def __init__(self):
        """Initialize Llama via Ollama"""
        self.base_url = LLAMA_CONFIG['base_url']
        self.model_name = LLAMA_CONFIG['model_name']
        self.temperature = LLAMA_CONFIG['temperature']
        self.max_tokens = LLAMA_CONFIG['max_tokens']
        
        # Check if Ollama is running
        self.available = self._check_ollama()
        
        if self.available and LANGCHAIN_AVAILABLE:
            self.llm = Ollama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature
            )
            print(f"✅ Llama initialized ({self.model_name})")
        else:
            print("⚠️  Llama not available (Ollama not running or LangChain not installed)")
            print("   Install: https://ollama.ai")
            print("   Run: ollama pull llama2")
            self.llm = None
    
    def _check_ollama(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate response using Llama
        
        Args:
            prompt: User query
            context: Additional context
            
        Returns:
            Generated response
        """
        if not self.available or not self.llm:
            return "⚠️  Llama not available. Using fallback response."
        
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        
        try:
            response = self.llm(full_prompt)
            return response
        except Exception as e:
            print(f"Error in Llama generation: {e}")
            return f"Error: {str(e)}"


class BioBERTEncoder:
    """
    BioBERT/ClinicalBERT encoder for medical text
    Generates embeddings for semantic search and matching
    """
    
    def __init__(self):
        """Initialize BioBERT encoder"""
        model_name = TEXT_ENCODER_CONFIG['model_name']
        
        print(f"Loading BioBERT: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.max_length = TEXT_ENCODER_CONFIG['max_length']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ BioBERT initialized on {self.device}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings


class BiomedCLIPProcessor:
    """
    BiomedCLIP for medical image understanding
    Handles clinical images: X-rays, CT, MRI, pathology
    """
    
    def __init__(self):
        """Initialize BiomedCLIP"""
        model_name = MEDCLIP_CONFIG['model_name']
        
        print(f"Loading BiomedCLIP: {model_name}...")
        
        try:
            # Try loading BiomedCLIP (may need special installation)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️  Could not load BiomedCLIP: {e}")
            print("   Using standard CLIP as fallback")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ BiomedCLIP initialized on {self.device}")
    
    def encode_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Encode image to embedding
        
        Args:
            image: Image path, array, or PIL Image
            
        Returns:
            Image embedding
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.cpu().numpy()
        
        return embedding
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding (for image-text matching)
        
        Args:
            text: Text description
            
        Returns:
            Text embedding
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            embedding = text_features.cpu().numpy()
        
        return embedding
    
    def analyze_image(self, image: Union[str, np.ndarray], 
                     candidate_labels: List[str]) -> Dict[str, float]:
        """
        Analyze image with candidate labels
        
        Args:
            image: Image to analyze
            candidate_labels: Possible conditions/findings
            
        Returns:
            Label probabilities
        """
        # Encode image
        image_emb = self.encode_image(image)
        
        # Encode labels
        label_embs = np.vstack([self.encode_text(label) for label in candidate_labels])
        
        # Compute similarities
        image_emb_norm = image_emb / np.linalg.norm(image_emb)
        label_embs_norm = label_embs / np.linalg.norm(label_embs, axis=1, keepdims=True)
        
        similarities = np.dot(label_embs_norm, image_emb_norm.T).flatten()
        
        # Softmax to probabilities
        probs = np.exp(similarities) / np.sum(np.exp(similarities))
        
        return dict(zip(candidate_labels, probs.tolist()))


class MultimodalFusionPipeline:
    """
    Orchestrates cross-modal processing using LangChain
    Integrates text and image understanding
    """
    
    def __init__(self):
        """Initialize multimodal pipeline"""
        print("\n" + "="*70)
        print("🚀 Initializing Multimodal Fusion Pipeline")
        print("="*70)
        
        # Initialize components
        self.llama = LlamaTextProcessor()
        self.biobert = BioBERTEncoder()
        self.biomedclip = BiomedCLIPProcessor()
        
        # Sentence transformer for semantic search
        try:
            self.sentence_encoder = SentenceTransformer(
                SENTENCE_TRANSFORMER_CONFIG['model_name']
            )
            print("✅ Sentence Transformer initialized")
        except Exception as e:
            print(f"⚠️  Sentence Transformer failed: {e}")
            self.sentence_encoder = None
        
        print("="*70)
        print("✅ Pipeline Ready")
        print("="*70 + "\n")
    
    def process_text_only(self, query: str, context: str) -> Dict:
        """
        Process text-only query
        
        Args:
            query: User question
            context: Clinical context
            
        Returns:
            Response with analysis
        """
        print(f"\n📝 Processing text-only query...")
        
        # Generate embedding for semantic search
        query_emb = self.biobert.encode(query)
        context_emb = self.biobert.encode(context)
        
        # Generate response with Llama
        prompt = PROMPT_TEMPLATES['text_only'].format(
            context=context,
            question=query
        )
        
        response = self.llama.generate_response(prompt)
        
        return {
            'mode': 'text_only',
            'query': query,
            'response': response,
            'query_embedding_shape': query_emb.shape,
            'context_embedding_shape': context_emb.shape
        }
    
    def process_image_only(self, image_path: str, image_type: str = 'xray') -> Dict:
        """
        Process image-only analysis
        
        Args:
            image_path: Path to medical image
            image_type: Type of image
            
        Returns:
            Analysis results
        """
        print(f"\n🖼️  Processing image-only analysis...")
        
        # Generate image embedding
        image_emb = self.biomedclip.encode_image(image_path)
        
        # Analyze with candidate conditions
        candidates = [
            'Normal chest X-ray',
            'Pneumonia',
            'COVID-19',
            'Pleural effusion',
            'Cardiomegaly',
            'Lung nodule'
        ]
        
        analysis = self.biomedclip.analyze_image(image_path, candidates)
        
        # Generate report with Llama
        findings = ", ".join([f"{k}: {v:.2%}" for k, v in sorted(
            analysis.items(), key=lambda x: x[1], reverse=True
        )[:3]])
        
        prompt = PROMPT_TEMPLATES['image_only'].format(
            image_type=image_type,
            context=f"Findings: {findings}"
        )
        
        response = self.llama.generate_response(prompt)
        
        return {
            'mode': 'image_only',
            'image_type': image_type,
            'findings': analysis,
            'response': response,
            'image_embedding_shape': image_emb.shape
        }
    
    def process_multimodal(self, query: str, text_context: str, 
                          image_path: str, image_type: str = 'xray') -> Dict:
        """
        Process multimodal query (text + image)
        
        Args:
            query: User question
            text_context: Clinical text
            image_path: Medical image path
            image_type: Type of image
            
        Returns:
            Integrated multimodal analysis
        """
        print(f"\n🔬 Processing multimodal query...")
        
        # Process text
        text_emb = self.biobert.encode(text_context)
        
        # Process image
        image_emb = self.biomedclip.encode_image(image_path)
        image_analysis = self.biomedclip.analyze_image(
            image_path,
            ['Normal', 'Abnormal - infection', 'Abnormal - tumor', 'Abnormal - other']
        )
        
        # Prepare integrated prompt
        image_findings = ", ".join([f"{k}: {v:.2%}" for k, v in sorted(
            image_analysis.items(), key=lambda x: x[1], reverse=True
        )[:2]])
        
        prompt = PROMPT_TEMPLATES['multimodal'].format(
            text_context=text_context,
            image_context=f"Image Analysis: {image_findings}",
            question=query
        )
        
        response = self.llama.generate_response(prompt)
        
        return {
            'mode': 'multimodal',
            'query': query,
            'text_embedding_shape': text_emb.shape,
            'image_embedding_shape': image_emb.shape,
            'image_findings': image_analysis,
            'response': response
        }
    
    def triage_assessment(self, patient_info: Dict) -> Dict:
        """
        Perform triage assessment
        
        Args:
            patient_info: Patient data (symptoms, vitals, history)
            
        Returns:
            Triage recommendation
        """
        print(f"\n🚨 Performing triage assessment...")
        
        prompt = PROMPT_TEMPLATES['triage'].format(
            patient_info=json.dumps(patient_info, indent=2),
            symptoms=patient_info.get('symptoms', 'Not provided'),
            vitals=patient_info.get('vitals', 'Not provided'),
            history=patient_info.get('history', 'Not provided')
        )
        
        response = self.llama.generate_response(prompt)
        
        return {
            'mode': 'triage',
            'patient_info': patient_info,
            'assessment': response
        }


if __name__ == "__main__":
    """Test cross-modal processing"""
    
    # Initialize pipeline
    pipeline = MultimodalFusionPipeline()
    
    # Test text-only
    result = pipeline.process_text_only(
        query="What is the likely diagnosis?",
        context="Patient with fever, cough, and bilateral lung infiltrates on chest X-ray."
    )
    
    print("\n" + "="*70)
    print("TEXT-ONLY RESULT:")
    print("="*70)
    print(result['response'])
    
    print("\n" + "="*70)
    print("Cross-modal processing test completed!")
    print("="*70)
