"""
Inference module for making predictions with the Multimodal Medical Assistant
Handles single and batch predictions on text and image data
"""

import os
import json
import torch
from PIL import Image
import yaml

# Import model and preprocessing
from models import MultimodalFusionModel, preprocess_text, get_image_transforms


def load_model(config_path='config.yaml', checkpoint_path=None):
    """
    Load trained model for inference.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint (optional)
        
    Returns:
        Loaded model in eval mode
    """
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'model': {'num_classes': 5, 'fusion_type': 'concatenation'}}
    
    # Initialize model
    model_config = config.get('model', {})
    model = MultimodalFusionModel(
        num_classes=model_config.get('num_classes', 5),
        fusion_type=model_config.get('fusion_type', 'concatenation'),
        hidden_dim=model_config.get('hidden_dim', 512),
        dropout=model_config.get('dropout', 0.3)
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded model from: {checkpoint_path}")
    else:
        print("⚠️  No checkpoint provided, using randomly initialized model")
    
    model.eval()
    return model


def run_prediction(text=None, image_path=None, config_path='config.yaml', 
                   output_path='./predictions.json', checkpoint_path=None):
    """
    Run prediction on text and/or image input.
    
    Args:
        text: Clinical text/notes (optional)
        image_path: Path to medical image (optional)
        config_path: Path to configuration file
        output_path: Where to save predictions
        checkpoint_path: Path to trained model checkpoint
        
    Returns:
        Prediction results dictionary
    """
    print("\n" + "="*70)
    print("🔮 RUNNING PREDICTION")
    print("="*70)
    
    # Load model
    model = load_model(config_path, checkpoint_path)
    
    # Prepare inputs
    results = {
        'input': {},
        'prediction': {},
        'confidence': 0.0,
        'model_info': 'Multimodal Medical Assistant'
    }
    
    # Process text input
    if text:
        print(f"\n📝 Text Input: {text[:100]}...")
        results['input']['text'] = text
        
        # Tokenize text
        text_encoded = preprocess_text(text, model.text_model.tokenizer)
        input_ids = text_encoded['input_ids']
        attention_mask = text_encoded['attention_mask']
    else:
        # Create dummy text input if not provided
        print("⚠️  No text provided, using placeholder")
        batch_size = 1
        input_ids = torch.zeros(batch_size, 128, dtype=torch.long)
        attention_mask = torch.ones(batch_size, 128, dtype=torch.long)
        results['input']['text'] = 'No text provided'
    
    # Process image input
    if image_path and os.path.exists(image_path):
        print(f"\n🖼️  Image Input: {image_path}")
        results['input']['image'] = image_path
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transform = get_image_transforms(is_training=False)
        image_tensor = transform(image).unsqueeze(0)
    else:
        # Create dummy image input if not provided
        if image_path:
            print(f"⚠️  Image not found: {image_path}, using placeholder")
        else:
            print("⚠️  No image provided, using placeholder")
        image_tensor = torch.randn(1, 3, 224, 224)
        results['input']['image'] = 'No image provided'
    
    # Run inference
    print("\n🧠 Running model inference...")
    with torch.no_grad():
        logits, fusion_features = model(input_ids, attention_mask, image_tensor)
        
        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Class labels (customize based on your task)
        class_labels = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Other']
        
        predicted_label = class_labels[predicted_class.item()] if predicted_class.item() < len(class_labels) else 'Unknown'
        
        results['prediction']['class'] = predicted_label
        results['prediction']['class_id'] = int(predicted_class.item())
        results['confidence'] = float(confidence.item())
        results['prediction']['probabilities'] = {
            class_labels[i]: float(probabilities[0, i].item()) 
            for i in range(min(len(class_labels), probabilities.shape[1]))
        }
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence.item():.2%}")
    print("\nClass Probabilities:")
    for label, prob in results['prediction']['probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    print("="*70)
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    """
    Direct execution for testing inference.
    """
    import sys
    
    # Example usage
    sample_text = "Patient presents with fever, cough, and shortness of breath. Chest X-ray shows bilateral infiltrates."
    
    run_prediction(
        text=sample_text,
        image_path=None,
        output_path='./sample_prediction.json'
    )
