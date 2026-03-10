"""
Multimodal Medical Assistant - Main Entry Point
Capstone Project CS[04]

Production-grade Multimodal Medical Assistant using:
- Llama 2/3 (via Ollama) for natural language understanding
- BiomedCLIP/MedCLIP for clinical image processing
- LangChain for cross-modal orchestration
- BioBERT/ClinicalBERT for medical text encoding
- OpenCV & Pydicom for medical image handling

Architecture:
    1. Multimodal Data Ingestion (Text: EHR, Images: DICOM/PNG/JPEG)
    2. Cross-Modal Processing (NLP + Vision fusion)
    3. LangChain Orchestration (Query routing and response generation)
    4. Clinical Decision Support Interface

Usage:
    API server (chatbot interface):
        python main.py --mode api --port 8000
    
    Multimodal query:
        python main.py --mode query --text "Interpret this X-ray" --image chest.dcm
    
    Process medical documents:
        python main.py --mode ingest --ehr-file report.txt --dicom-dir ./scans/
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to run the Multimodal Medical Assistant.
    
    Modes:
        - train: Train the multimodal model
        - api: Start FastAPI server for inference
        - predict: Run batch prediction on text and image
        - evaluate: Evaluate model performance
    """
    parser = argparse.ArgumentParser(
        description='Multimodal Medical Assistant - AI-powered clinical decision support system'
    )
    
    # Core arguments
    parser.add_argument(
        '--mode',
        type=str,
        default='api',
        choices=['train', 'api', 'predict', 'evaluate', 'demo', 'query', 'ingest', 'interactive'],
        help='Operation mode: train, api, predict, evaluate, demo, query (multimodal query), ingest (data ingestion), interactive (chatbot)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    # API mode arguments
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API host address (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API port number (default: 8000)'
    )
    
    # Prediction mode arguments
    parser.add_argument(
        '--text',
        type=str,
        help='Clinical text/notes for prediction or query'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to medical image (DICOM, PNG, JPEG) for prediction or query'
    )
    
    parser.add_argument(
        '--context',
        type=str,
        help='Additional clinical context for multimodal query'
    )
    
    parser.add_argument(
        '--ehr-file',
        type=str,
        help='Path to EHR/clinical text file for ingestion'
    )
    
    parser.add_argument(
        '--dicom-dir',
        type=str,
        help='Path to directory containing DICOM images for ingestion'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./predictions.json',
        help='Output file for predictions (default: ./predictions.json)'
    )
    
    # Environment variable for API keys (if needed)
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for external services (optional, can also use env var API_KEY)'
    )
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv('API_KEY')
    
    print("="*70)
    print("🏥 MULTIMODAL MEDICAL ASSISTANT - Capstone Project CS[04]")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {args.config}")
    print("="*70)
    print()
    
    # Execute based on mode
    if args.mode == 'train':
        print("📚 Starting training mode...")
        from train import train_model
        train_model(args.config)
        
    elif args.mode == 'api':
        print("🚀 Starting API server...")
        print(f"Server will be available at: http://{args.host}:{args.port}")
        print(f"API documentation: http://{args.host}:{args.port}/docs")
        print()
        
        # Import and run FastAPI server
        import uvicorn
        from api import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level='info'
        )
        
    elif args.mode == 'predict':
        print("🔮 Starting prediction mode...")
        from inference import run_prediction
        
        # Validate inputs
        if not args.text and not args.image:
            print("❌ Error: At least one of --text or --image must be provided")
            sys.exit(1)
        
        # Run prediction
        result = run_prediction(
            text=args.text,
            image_path=args.image,
            config_path=args.config,
            output_path=args.output
        )
        
        print(f"\n✅ Prediction complete! Results saved to: {args.output}")
        
    elif args.mode == 'evaluate':
        print("📊 Starting evaluation mode...")
        from evaluate import evaluate_model
        evaluate_model(args.config)
        
    elif args.mode == 'demo':
        print("🎯 Starting demo mode...")
        from demo import run_demo
        run_demo(args.config)
    
    elif args.mode == 'query':
        print("🔬 Starting multimodal query mode...")
        from cross_modal import MultimodalFusionPipeline
        
        # Validate inputs
        if not args.text and not args.image:
            print("❌ Error: At least one of --text or --image must be provided")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = MultimodalFusionPipeline()
        
        # Process query
        if args.text and args.image:
            print("\n📊 Processing MULTIMODAL query (text + image)...")
            result = pipeline.process_multimodal(
                query=args.text,
                text_context=args.context or "",
                image_path=args.image
            )
        elif args.image:
            print("\n🖼️  Processing IMAGE-ONLY query...")
            result = pipeline.process_image_only(args.image)
        else:
            print("\n📝 Processing TEXT-ONLY query...")
            result = pipeline.process_text_only(
                query=args.text,
                context=args.context or ""
            )
        
        # Display results
        print("\n" + "="*70)
        print("QUERY RESULTS")
        print("="*70)
        print(f"Mode: {result['mode']}")
        print(f"\nResponse:\n{result['response']}")
        print("="*70)
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {args.output}")
    
    elif args.mode == 'ingest':
        print("📥 Starting data ingestion mode...")
        from ingestion import MedicalTextIngestion, MedicalImageIngestion
        
        results = {}
        
        # Ingest text data
        if args.ehr_file:
            print(f"\n📄 Processing EHR file: {args.ehr_file}")
            text_ingestion = MedicalTextIngestion()
            
            with open(args.ehr_file, 'r') as f:
                text_data = f.read()
            
            processed = text_ingestion.process_ehr_text(text_data)
            results['text'] = {
                'original_length': len(text_data),
                'processed_length': len(processed['deidentified_text']),
                'phi_removed': processed['phi_removed']
            }
            
            print(f"✅ Text processed: {results['text']['phi_removed']} PHI entities removed")
        
        # Ingest image data
        if args.dicom_dir:
            print(f"\n🖼️  Processing DICOM directory: {args.dicom_dir}")
            image_ingestion = MedicalImageIngestion()
            
            import glob
            dicom_files = glob.glob(os.path.join(args.dicom_dir, '*.dcm'))
            
            processed_count = 0
            for dicom_path in dicom_files[:5]:  # Limit to 5 for demo
                try:
                    processed = image_ingestion.process_dicom(dicom_path)
                    if processed is not None:
                        processed_count += 1
                except Exception as e:
                    print(f"⚠️  Error processing {dicom_path}: {e}")
            
            results['images'] = {
                'total_files': len(dicom_files),
                'processed': processed_count
            }
            
            print(f"✅ Images processed: {processed_count}/{len(dicom_files)}")
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Ingestion results saved to: {args.output}")
    
    elif args.mode == 'interactive':
        print("🤖 Starting interactive chatbot mode...")
        from clinical_interface import interactive_demo
        interactive_demo()
    
    print()
    print("="*70)
    print("✨ Execution completed successfully!")
    print("="*70)


if __name__ == "__main__":
    """
    Entry point when running the script directly.
    Handles all command-line arguments and routes to appropriate functions.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
