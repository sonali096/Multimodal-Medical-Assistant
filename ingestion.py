"""
Multimodal Data Ingestion Module
Handles upload, processing, and de-identification of medical text and images

Components:
1. Medical Text Ingestion (EHR, discharge summaries, clinical notes)
2. Medical Image Ingestion (DICOM, PNG, JPEG from radiology/pathology)
3. De-identification & Privacy Compliance (HIPAA)
4. Data Validation and Quality Checks
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Image processing
import cv2
import numpy as np
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("⚠️  pydicom not available. Install with: pip install pydicom")

from PIL import Image

# NLP for de-identification
import spacy

# Configuration
from config import (
    STORAGE_CONFIG, PRIVACY_CONFIG, IMAGE_PROCESSING_CONFIG,
    TEXT_PROCESSING_CONFIG
)


class MedicalTextIngestion:
    """
    Handles ingestion and processing of medical text documents
    Includes de-identification and entity extraction
    """
    
    def __init__(self):
        """Initialize text ingestion with NLP models"""
        self.upload_dir = STORAGE_CONFIG['upload_dir']
        self.processed_dir = STORAGE_CONFIG['processed_dir']
        
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Load spaCy for de-identification
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("⚠️  Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        print("✅ Medical Text Ingestion initialized")
    
    def deidentify_text(self, text: str) -> Tuple[str, Dict]:
        """
        Remove Protected Health Information (PHI) from text
        
        Args:
            text: Raw medical text
            
        Returns:
            (deidentified_text, phi_mapping): Cleaned text and PHI entities found
        """
        if not PRIVACY_CONFIG['enable_deidentification']:
            return text, {}
        
        doc = self.nlp(text)
        phi_mapping = {}
        deidentified = text
        
        # Track PHI entities
        phi_types = TEXT_PROCESSING_CONFIG['phi_types']
        
        # Remove person names
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                replacement = f"[PATIENT_{len(phi_mapping)}]"
                phi_mapping[replacement] = ent.text
                deidentified = deidentified.replace(ent.text, replacement)
        
        # Remove dates (keep only year for age calculation)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        for match in re.finditer(date_pattern, deidentified):
            replacement = "[DATE]"
            phi_mapping[replacement] = match.group()
            deidentified = deidentified.replace(match.group(), replacement)
        
        # Remove phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, deidentified):
            replacement = "[PHONE]"
            phi_mapping[replacement] = match.group()
            deidentified = deidentified.replace(match.group(), replacement)
        
        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, deidentified):
            replacement = "[EMAIL]"
            phi_mapping[replacement] = match.group()
            deidentified = deidentified.replace(match.group(), replacement)
        
        # Remove SSN
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, deidentified):
            replacement = "[SSN]"
            phi_mapping[replacement] = match.group()
            deidentified = deidentified.replace(match.group(), replacement)
        
        return deidentified, phi_mapping
    
    def ingest_text(self, text: str, document_type: str = 'clinical_note',
                   metadata: Optional[Dict] = None) -> Dict:
        """
        Ingest medical text document
        
        Args:
            text: Medical text content
            document_type: Type of document (clinical_note, ehr, discharge_summary)
            metadata: Additional metadata
            
        Returns:
            Processing result with document ID and statistics
        """
        print(f"\n📝 Ingesting {document_type}...")
        
        # Generate unique document ID
        doc_id = hashlib.sha256(
            f"{text[:100]}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # De-identify if enabled
        deidentified_text, phi_mapping = self.deidentify_text(text)
        
        # Extract medical entities
        doc = self.nlp(deidentified_text)
        entities = {
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
            'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
            'orgs': [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        }
        
        # Prepare output
        result = {
            'document_id': doc_id,
            'document_type': document_type,
            'original_length': len(text),
            'processed_length': len(deidentified_text),
            'phi_removed': len(phi_mapping),
            'entities_found': sum(len(v) for v in entities.values()),
            'timestamp': datetime.now().isoformat(),
            'deidentified': PRIVACY_CONFIG['enable_deidentification']
        }
        
        # Save processed document
        output_file = os.path.join(self.processed_dir, f'{doc_id}_text.json')
        with open(output_file, 'w') as f:
            json.dump({
                'document_id': doc_id,
                'text': deidentified_text,
                'entities': entities,
                'metadata': metadata or {},
                'phi_count': len(phi_mapping)
            }, f, indent=2)
        
        print(f"✅ Text ingested: {doc_id}")
        print(f"   - Original: {result['original_length']} chars")
        print(f"   - Processed: {result['processed_length']} chars")
        print(f"   - PHI removed: {result['phi_removed']} items")
        
        return result


class MedicalImageIngestion:
    """
    Handles ingestion and processing of medical images
    Supports DICOM, PNG, JPEG formats
    """
    
    def __init__(self):
        """Initialize image ingestion with OpenCV and pydicom"""
        self.upload_dir = STORAGE_CONFIG['upload_dir']
        self.processed_dir = STORAGE_CONFIG['processed_dir']
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.supported_formats = IMAGE_PROCESSING_CONFIG['supported_formats']
        
        print("✅ Medical Image Ingestion initialized")
        if not PYDICOM_AVAILABLE:
            print("   ⚠️  DICOM support limited (pydicom not installed)")
    
    def process_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Process DICOM medical image
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            (image_array, metadata): Processed image and DICOM metadata
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM processing")
        
        # Read DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Extract metadata
        metadata = {
            'patient_id': str(getattr(dicom, 'PatientID', 'Unknown')),
            'study_date': str(getattr(dicom, 'StudyDate', 'Unknown')),
            'modality': str(getattr(dicom, 'Modality', 'Unknown')),
            'body_part': str(getattr(dicom, 'BodyPartExamined', 'Unknown')),
            'rows': int(dicom.Rows),
            'columns': int(dicom.Columns)
        }
        
        # Get pixel array
        image = dicom.pixel_array.astype(float)
        
        # Apply windowing for better visualization
        window_center = IMAGE_PROCESSING_CONFIG['dicom_window_center']
        window_width = IMAGE_PROCESSING_CONFIG['dicom_window_width']
        
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image = np.clip(image, img_min, img_max)
        
        # Normalize to 0-255
        image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # De-identify metadata
        if PRIVACY_CONFIG['enable_deidentification']:
            metadata['patient_id'] = '[PATIENT_ID]'
            metadata['study_date'] = '[DATE]'
        
        return image, metadata
    
    def process_standard_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Process standard image formats (PNG, JPEG)
        
        Args:
            image_path: Path to image file
            
        Returns:
            (image_array, metadata): Processed image and metadata
        """
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        metadata = {
            'format': Path(image_path).suffix.lower(),
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        
        return image, metadata
    
    def ingest_image(self, image_path: str, image_type: str = 'xray',
                    metadata: Optional[Dict] = None) -> Dict:
        """
        Ingest medical image
        
        Args:
            image_path: Path to image file
            image_type: Type of image (xray, ct, mri, pathology)
            metadata: Additional metadata
            
        Returns:
            Processing result with image ID and statistics
        """
        print(f"\n🖼️  Ingesting {image_type} image...")
        
        file_ext = Path(image_path).suffix.lower().replace('.', '')
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Generate unique image ID
        img_id = hashlib.sha256(
            f"{image_path}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Process based on format
        if file_ext in ['dcm', 'dicom']:
            image, img_metadata = self.process_dicom(image_path)
        else:
            image, img_metadata = self.process_standard_image(image_path)
        
        # Resize if needed
        target_size = IMAGE_PROCESSING_CONFIG['resize']
        if image.shape != target_size:
            image = cv2.resize(image, target_size)
        
        # Normalize
        if IMAGE_PROCESSING_CONFIG['normalize']:
            image = image.astype(np.float32) / 255.0
        
        # Save processed image
        output_image_path = os.path.join(self.processed_dir, f'{img_id}_image.png')
        cv2.imwrite(output_image_path, (image * 255).astype(np.uint8))
        
        # Save metadata
        output_meta_path = os.path.join(self.processed_dir, f'{img_id}_meta.json')
        with open(output_meta_path, 'w') as f:
            json.dump({
                'image_id': img_id,
                'image_type': image_type,
                'original_path': image_path,
                'processed_path': output_image_path,
                'metadata': {**img_metadata, **(metadata or {})},
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        result = {
            'image_id': img_id,
            'image_type': image_type,
            'format': file_ext,
            'shape': image.shape,
            'saved_path': output_image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Image ingested: {img_id}")
        print(f"   - Type: {image_type}")
        print(f"   - Format: {file_ext}")
        print(f"   - Shape: {image.shape}")
        
        return result


# ==================== CONVENIENCE FUNCTIONS ====================

def ingest_medical_data(text: Optional[str] = None,
                       image_path: Optional[str] = None,
                       document_type: str = 'clinical_note',
                       image_type: str = 'xray',
                       metadata: Optional[Dict] = None) -> Dict:
    """
    Unified interface for ingesting both text and image data
    
    Args:
        text: Medical text content
        image_path: Path to medical image
        document_type: Type of text document
        image_type: Type of medical image
        metadata: Additional metadata
        
    Returns:
        Combined ingestion results
    """
    results = {
        'text_result': None,
        'image_result': None,
        'status': 'success'
    }
    
    try:
        if text:
            text_ingestor = MedicalTextIngestion()
            results['text_result'] = text_ingestor.ingest_text(
                text, document_type, metadata
            )
        
        if image_path:
            image_ingestor = MedicalImageIngestion()
            results['image_result'] = image_ingestor.ingest_image(
                image_path, image_type, metadata
            )
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        print(f"❌ Error during ingestion: {e}")
    
    return results


if __name__ == "__main__":
    """Test ingestion modules"""
    
    # Test text ingestion
    sample_text = """
    Patient: John Doe
    DOB: 01/15/1975
    Phone: 555-123-4567
    Email: john.doe@email.com
    
    Chief Complaint: Persistent cough and fever for 5 days.
    
    History: Patient presents with productive cough, fever (101°F), 
    and shortness of breath. Denies chest pain. No recent travel.
    
    Assessment: Likely community-acquired pneumonia.
    Plan: Chest X-ray, CBC, start empiric antibiotics.
    """
    
    text_ing = MedicalTextIngestion()
    text_result = text_ing.ingest_text(sample_text, 'clinical_note')
    
    print("\n" + "="*70)
    print("Ingestion test completed!")
    print("="*70)
