"""
Demo module for quick testing of the Multimodal Medical Assistant
Runs sample predictions to demonstrate functionality
"""

import os


def run_demo(config_path='config.yaml'):
    """
    Run a demonstration of the system with sample data.
    
    Args:
        config_path: Path to configuration file
    """
    print("\n" + "="*70)
    print("🎯 MULTIMODAL MEDICAL ASSISTANT - DEMO MODE")
    print("="*70)
    
    from inference import run_prediction
    
    # Sample clinical scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Pneumonia Suspect',
            'text': 'Patient presents with high fever (102°F), productive cough with yellow sputum, and difficulty breathing. Physical exam reveals crackles in lower lung fields. Oxygen saturation 92% on room air.',
            'image': None
        },
        {
            'name': 'Scenario 2: Routine Checkup',
            'text': 'Annual health screening. Patient reports no complaints. Vital signs within normal limits. No abnormalities detected on physical examination.',
            'image': None
        },
        {
            'name': 'Scenario 3: COVID-19 Symptoms',
            'text': 'Patient reports loss of taste and smell, dry cough, fever, and fatigue for 3 days. Recent exposure to confirmed COVID-19 case. Chest X-ray shows bilateral ground-glass opacities.',
            'image': None
        }
    ]
    
    print("\n🔬 Running sample clinical scenarios...")
    print()
    
    results_summary = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"📋 {scenario['name']}")
        print(f"{'='*70}")
        print(f"Clinical Notes: {scenario['text'][:150]}...")
        
        # Run prediction
        output_file = f'./demo_result_{i}.json'
        result = run_prediction(
            text=scenario['text'],
            image_path=scenario['image'],
            config_path=config_path,
            output_path=output_file
        )
        
        results_summary.append({
            'scenario': scenario['name'],
            'prediction': result['prediction']['class'],
            'confidence': result['confidence']
        })
        
        print(f"\n✅ Scenario {i} completed")
    
    # Summary
    print("\n" + "="*70)
    print("📊 DEMO SUMMARY")
    print("="*70)
    for i, summary in enumerate(results_summary, 1):
        print(f"\n{i}. {summary['scenario']}")
        print(f"   Prediction: {summary['prediction']}")
        print(f"   Confidence: {summary['confidence']:.2%}")
    
    print("\n" + "="*70)
    print("✨ Demo completed successfully!")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Train with real medical data")
    print("   2. Fine-tune the model")
    print("   3. Deploy via API: python main.py --mode api")
    print()


if __name__ == "__main__":
    """
    Direct execution for demo.
    """
    run_demo()
