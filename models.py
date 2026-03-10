"""
Model architectures (simplified for flat structure)
Contains text, image, and multimodal fusion models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models
from torchvision import transforms


# ==================== TEXT MODEL ====================

class MedicalTextModel(nn.Module):
    """Medical text encoder using BioBERT/ClinicalBERT"""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", num_classes=5, dropout=0.3):
        super(MedicalTextModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output


def preprocess_text(text, tokenizer, max_length=512):
    """Preprocess medical text for model input"""
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded


# ==================== IMAGE MODEL ====================

class MedicalImageModel(nn.Module):
    """Medical image encoder using ResNet/DenseNet"""
    
    def __init__(self, model_name="resnet50", num_classes=5, pretrained=True, dropout=0.3):
        super(MedicalImageModel, self).__init__()
        
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.feature_dim = feature_dim
        
    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


def get_image_transforms(img_size=224, is_training=False):
    """Get image preprocessing transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ==================== MULTIMODAL FUSION MODEL ====================

class MultimodalFusionModel(nn.Module):
    """Multimodal fusion combining text and image"""
    
    def __init__(self, text_model_name="emilyalsentzer/Bio_ClinicalBERT",
                 image_model_name="resnet50", num_classes=5,
                 fusion_type="concatenation", hidden_dim=512, dropout=0.3):
        super(MultimodalFusionModel, self).__init__()
        
        self.text_model = MedicalTextModel(model_name=text_model_name, num_classes=num_classes)
        self.image_model = MedicalImageModel(model_name=image_model_name, num_classes=num_classes)
        
        text_dim = self.text_model.bert.config.hidden_size
        image_dim = self.image_model.feature_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concatenation":
            self.fusion = nn.Sequential(
                nn.Linear(text_dim + image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            final_dim = hidden_dim // 2
        else:
            final_dim = hidden_dim
        
        self.classifier = nn.Linear(final_dim, num_classes)
        
    def forward(self, text_input_ids, text_attention_mask, images):
        _, text_embeddings = self.text_model(text_input_ids, text_attention_mask)
        _, image_embeddings = self.image_model(images)
        
        if self.fusion_type == "concatenation":
            combined = torch.cat([text_embeddings, image_embeddings], dim=1)
            fusion_features = self.fusion(combined)
        else:
            fusion_features = text_embeddings + image_embeddings
        
        logits = self.classifier(fusion_features)
        return logits, fusion_features
