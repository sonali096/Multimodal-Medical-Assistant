"""
Clinical Assistant Interface
Interactive chatbot and dashboard for multimodal medical queries

Features:
1. Conversational interface with Llama
2. Multimodal query processing (text + images)
3. Diagnostic suggestions and evidence
4. Triage assessment
5. Medical knowledge retrieval
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Cross-modal processing
from cross_modal import MultimodalFusionPipeline
from ingestion import MedicalTextIngestion, MedicalImageIngestion

# Vector store for knowledge retrieval
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False
    print("⚠️  Vector store not available. Install: pip install faiss-cpu langchain")


class MedicalKnowledgeBase:
    """
    Vector database for medical knowledge retrieval
    Supports semantic search over clinical guidelines, research, cases
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize knowledge base
        
        Args:
            index_path: Path to saved FAISS index
        """
        self.index_path = index_path or './data/knowledge_base.faiss'
        
        if VECTORSTORE_AVAILABLE:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name='emilyalsentzer/Bio_ClinicalBERT'
            )
            
            # Load or create index
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(self.index_path, self.embeddings)
                print(f"✅ Knowledge base loaded from {self.index_path}")
            else:
                self.vectorstore = None
                print("⚠️  No knowledge base found. Use add_documents() to create one.")
        else:
            self.vectorstore = None
            print("⚠️  Vector store not available")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to knowledge base
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
        """
        if not VECTORSTORE_AVAILABLE:
            print("⚠️  Vector store not available")
            return
        
        from langchain.schema import Document
        
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        
        # Save index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        
        print(f"✅ Added {len(texts)} documents to knowledge base")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Semantic search in knowledge base
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return [(doc.page_content, score) for doc, score in results]


class ClinicalChatbot:
    """
    Interactive clinical assistant chatbot
    Handles conversational queries with context
    """
    
    def __init__(self):
        """Initialize chatbot"""
        print("\n" + "="*70)
        print("🤖 Initializing Clinical Chatbot")
        print("="*70)
        
        # Initialize pipeline
        self.pipeline = MultimodalFusionPipeline()
        
        # Initialize knowledge base
        self.knowledge_base = MedicalKnowledgeBase()
        
        # Initialize data processors
        self.text_ingestion = MedicalTextIngestion()
        self.image_ingestion = MedicalImageIngestion()
        
        # Conversation history
        self.conversation_history: List[Dict] = []
        
        print("="*70)
        print("✅ Chatbot Ready")
        print("="*70 + "\n")
    
    def process_query(self, query: str, image_path: Optional[str] = None,
                     patient_context: Optional[str] = None) -> Dict:
        """
        Process user query
        
        Args:
            query: User question
            image_path: Optional medical image
            patient_context: Optional clinical context
            
        Returns:
            Response with analysis
        """
        timestamp = datetime.now().isoformat()
        
        # De-identify context if provided
        if patient_context:
            patient_context = self.text_ingestion.deidentify_text(patient_context)
        
        # Determine query mode
        if image_path and patient_context:
            # Multimodal query
            result = self.pipeline.process_multimodal(
                query=query,
                text_context=patient_context,
                image_path=image_path
            )
        elif image_path:
            # Image-only query
            result = self.pipeline.process_image_only(image_path)
        elif patient_context:
            # Text-only query
            result = self.pipeline.process_text_only(query, patient_context)
        else:
            # General query - use knowledge base
            kb_results = self.knowledge_base.search(query, k=3)
            context = "\n".join([doc for doc, _ in kb_results]) if kb_results else ""
            result = self.pipeline.process_text_only(query, context)
        
        # Add to conversation history
        conversation_entry = {
            'timestamp': timestamp,
            'query': query,
            'image_path': image_path,
            'context': patient_context,
            'result': result
        }
        self.conversation_history.append(conversation_entry)
        
        return result
    
    def get_diagnostic_suggestions(self, symptoms: List[str], 
                                   findings: Dict) -> List[Dict]:
        """
        Generate diagnostic suggestions
        
        Args:
            symptoms: List of symptoms
            findings: Examination findings
            
        Returns:
            List of diagnostic hypotheses with evidence
        """
        # Prepare clinical summary
        summary = f"Symptoms: {', '.join(symptoms)}\nFindings: {json.dumps(findings)}"
        
        # Search knowledge base for similar cases
        kb_results = self.knowledge_base.search(summary, k=5)
        
        # Generate suggestions with Llama
        context = "\n".join([doc for doc, _ in kb_results]) if kb_results else ""
        
        query = f"Based on these symptoms and findings, what are the most likely diagnoses?"
        
        result = self.pipeline.process_text_only(query, summary + "\n\n" + context)
        
        # Parse response into structured suggestions
        # (In production, use more sophisticated parsing)
        suggestions = [
            {
                'diagnosis': 'Generated by LLM',
                'confidence': 0.0,
                'evidence': result['response'],
                'references': [doc for doc, _ in kb_results[:3]]
            }
        ]
        
        return suggestions
    
    def perform_triage(self, patient_data: Dict) -> Dict:
        """
        Perform triage assessment
        
        Args:
            patient_data: Patient information
            
        Returns:
            Triage result with urgency level
        """
        # De-identify patient data
        if 'history' in patient_data:
            patient_data['history'] = self.text_ingestion.deidentify_text(
                patient_data['history']
            )
        
        # Perform triage
        result = self.pipeline.triage_assessment(patient_data)
        
        # Add to history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'triage',
            'patient_data': patient_data,
            'result': result
        })
        
        return result
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of conversation
        
        Returns:
            Formatted conversation summary
        """
        if not self.conversation_history:
            return "No conversation history"
        
        summary = f"Conversation Summary ({len(self.conversation_history)} exchanges)\n"
        summary += "=" * 70 + "\n\n"
        
        for i, entry in enumerate(self.conversation_history, 1):
            summary += f"{i}. [{entry['timestamp']}]\n"
            summary += f"   Query: {entry.get('query', entry.get('type', 'Unknown'))}\n"
            
            if 'result' in entry:
                result = entry['result']
                if 'response' in result:
                    response = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
                    summary += f"   Response: {response}\n"
            
            summary += "\n"
        
        return summary


class ClinicalDashboard:
    """
    Dashboard interface for clinical workflows
    Provides structured access to assistant capabilities
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.chatbot = ClinicalChatbot()
        self.active_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'queries': []
        }
    
    def upload_patient_data(self, text_data: Optional[str] = None,
                           image_path: Optional[str] = None) -> Dict:
        """
        Upload and process patient data
        
        Args:
            text_data: Clinical notes/reports
            image_path: Medical image
            
        Returns:
            Processing results
        """
        results = {}
        
        # Process text
        if text_data:
            processed_text = self.chatbot.text_ingestion.process_ehr_text(text_data)
            results['text'] = processed_text
        
        # Process image
        if image_path:
            if image_path.endswith('.dcm'):
                processed_image = self.chatbot.image_ingestion.process_dicom(image_path)
            else:
                processed_image = self.chatbot.image_ingestion.process_standard_image(image_path)
            results['image'] = {
                'path': image_path,
                'processed': processed_image is not None
            }
        
        return results
    
    def query_assistant(self, query: str, image_path: Optional[str] = None,
                       context: Optional[str] = None) -> Dict:
        """
        Query the clinical assistant
        
        Args:
            query: User question
            image_path: Optional image
            context: Optional context
            
        Returns:
            Assistant response
        """
        result = self.chatbot.process_query(query, image_path, context)
        
        # Log query
        self.active_session['queries'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result': result
        })
        
        return result
    
    def get_session_report(self) -> str:
        """
        Generate session report
        
        Returns:
            Formatted session report
        """
        report = f"Clinical Session Report\n"
        report += f"Session ID: {self.active_session['session_id']}\n"
        report += f"Started: {self.active_session['start_time']}\n"
        report += f"Queries: {len(self.active_session['queries'])}\n"
        report += "=" * 70 + "\n\n"
        
        report += self.chatbot.get_conversation_summary()
        
        return report
    
    def export_session(self, output_path: str):
        """
        Export session data to JSON
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.active_session, f, indent=2)
        
        print(f"✅ Session exported to {output_path}")


def interactive_demo():
    """Interactive demo of clinical interface"""
    
    print("\n" + "="*70)
    print("🏥 MULTIMODAL MEDICAL ASSISTANT - INTERACTIVE DEMO")
    print("="*70)
    
    # Initialize dashboard
    dashboard = ClinicalDashboard()
    
    # Demo queries
    demos = [
        {
            'name': 'Text-Only Query',
            'query': 'What are the treatment options for community-acquired pneumonia?',
            'context': 'Patient is a 45-year-old with fever, cough, and chest pain for 3 days.'
        },
        {
            'name': 'Knowledge Retrieval',
            'query': 'What is the CURB-65 score and how is it calculated?',
            'context': None
        },
        {
            'name': 'Triage Assessment',
            'type': 'triage',
            'patient_data': {
                'symptoms': ['fever', 'cough', 'shortness of breath'],
                'vitals': {
                    'temperature': 38.5,
                    'heart_rate': 110,
                    'respiratory_rate': 24,
                    'oxygen_saturation': 92
                },
                'history': 'John Smith, 65 years old, has diabetes and hypertension.'
            }
        }
    ]
    
    # Run demos
    for i, demo in enumerate(demos, 1):
        print(f"\n{'='*70}")
        print(f"DEMO {i}: {demo['name']}")
        print("="*70)
        
        if demo.get('type') == 'triage':
            result = dashboard.chatbot.perform_triage(demo['patient_data'])
            print("\n📊 TRIAGE RESULT:")
            print(result['assessment'])
        else:
            result = dashboard.query_assistant(demo['query'], context=demo.get('context'))
            print(f"\n❓ Query: {demo['query']}")
            if demo.get('context'):
                print(f"📋 Context: {demo['context']}")
            print(f"\n💬 Response:")
            print(result['response'])
    
    # Session report
    print("\n" + "="*70)
    print("📈 SESSION REPORT")
    print("="*70)
    print(dashboard.get_session_report())
    
    # Export session
    export_path = f"./data/session_{dashboard.active_session['session_id']}.json"
    os.makedirs('./data', exist_ok=True)
    dashboard.export_session(export_path)
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    """Run interactive demo"""
    interactive_demo()
