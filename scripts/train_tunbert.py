"""
Script d'entraînement TunBERT avec MLflow - MLOps Election
Fine-tuning du modèle TunBERT pour la classification de sentiments
"""

import os
import sys
import pandas as pd
import pickle
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# PyTorch & Transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)

# MLflow & DagsHub
import mlflow
import dagshub

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

PROCESSOR_DIR = BASE_DIR / 'processors'
MODELS_DIR = BASE_DIR / 'models'
TUNBERT_DIR = MODELS_DIR / 'tunbert_final_model'
TUNBERT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🚀 ENTRAÎNEMENT TUNBERT - MLOps Election")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Répertoire: {BASE_DIR}")
print()

# ============================================================================
# Configuration MLflow & GPU
# ============================================================================

def setup_mlflow():
    """Configure MLflow tracking avec DagsHub"""
    DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'rahmmaakhlefa')
    DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'tunsent-mlops')
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI',
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )
    
    print("🔧 Configuration MLflow & DagsHub")
    print("-" * 80)
    
    try:
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
        print(f"✅ DagsHub initialisé: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
    except Exception as e:
        print(f"⚠️  DagsHub warning: {e}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentiment_classification_tunisian3")
    
    print(f"✅ MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print()

def setup_device():
    """Configure GPU/CPU"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🎮 Configuration GPU/CPU")
    print("-" * 80)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️  GPU non disponible - entraînement sur CPU (lent)")
    
    print()
    return device

# ============================================================================
# Chargement des données
# ============================================================================

def load_cleaned_texts():
    """Charge les textes nettoyés"""
    print("📦 Chargement des textes nettoyés")
    print("-" * 80)
    
    texts_path = PROCESSOR_DIR / 'cleaned_texts.pkl'
    if not texts_path.exists():
        raise FileNotFoundError(
            f"Textes nettoyés non trouvés: {texts_path}\n"
            "Exécutez d'abord: python scripts/preprocess.py"
        )
    
    with open(texts_path, 'rb') as f:
        data = pickle.load(f)
    
    comments = data['cleaned']
    labels = data['labels']
    
    print(f"✅ Textes chargés: {len(comments)} commentaires")
    print(f"   Distribution: {pd.Series(labels).value_counts().to_dict()}")
    print()
    
    return comments, labels

def create_datasets(comments, labels, test_size=0.15, val_size=0.15, random_state=42):
    """Crée les datasets train/val/test stratifiés"""
    print("✂️  Split des données (70-15-15)")
    print("-" * 80)
    
    df = pd.DataFrame({'comments': comments, 'target': labels})
    
    # Train/temp split
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), 
        random_state=random_state, stratify=df['target']
    )
    
    # Val/test split
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, 
        random_state=random_state, stratify=temp_df['target']
    )
    
    print(f"✅ Split terminé:")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print()
    
    # Convertir en HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['comments', 'target']])
    val_dataset = Dataset.from_pandas(val_df[['comments', 'target']])
    test_dataset = Dataset.from_pandas(test_df[['comments', 'target']])
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# Modèle TunBERT
# ============================================================================

def load_tunbert_model(device):
    """Charge TunBERT et le tokenizer"""
    print("🔄 Chargement de TunBERT")
    print("-" * 80)
    
    model_name = "tunis-ai/TunBERT"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=2
    )
    
    # Déplacer sur GPU
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ TunBERT chargé: {model_name}")
    print(f"   Paramètres: {num_params:,}")
    print(f"   Device: {device}")
    print()
    
    return model, tokenizer

def tokenize_datasets(train_dataset, val_dataset, test_dataset, tokenizer, max_length=128):
    """Tokenize tous les datasets"""
    print("🔢 Tokenization des datasets")
    print("-" * 80)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['comments'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Renommer target -> labels
    train_dataset = train_dataset.rename_column("target", "labels")
    val_dataset = val_dataset.rename_column("target", "labels")
    test_dataset = test_dataset.rename_column("target", "labels")
    
    # Format PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"✅ Tokenization terminée (max_length={max_length})")
    print()
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# Entraînement avec MLflow
# ============================================================================

def compute_metrics(pred):
    """Calcule les métriques pour le Trainer"""
    labels = pred.label_ids
    predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_tunbert(model, tokenizer, train_dataset, val_dataset, test_dataset, device, 
                  epochs=7, batch_size=8, learning_rate=2e-5):
    """Entraîne TunBERT avec MLflow tracking"""
    print("🚀 ENTRAÎNEMENT TUNBERT AVEC MLFLOW")
    print("="*80)
    
    # Démarrer le run MLflow
    with mlflow.start_run(run_name="TunBERT_Transformer"):
        
        # Log hyperparamètres
        mlflow.log_param("model_type", "TunBERT")
        mlflow.log_param("model_name", "tunis-ai/TunBERT")
        mlflow.log_param("num_labels", 2)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_length", 128)
        mlflow.log_param("device", str(device))
        mlflow.log_param("fp16", torch.cuda.is_available())
        
        # Configuration du training
        training_args = TrainingArguments(
            output_dir=str(BASE_DIR / "tunbert_finetuned"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=str(BASE_DIR / "logs"),
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Créer le Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Entraînement
        print("\n⏳ Entraînement en cours (peut prendre plusieurs minutes)...\n")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Log métriques d'entraînement
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("training_loss", train_result.training_loss)
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   Training loss: {train_result.training_loss:.4f}")
        print(f"   Temps: {training_time:.1f}s ({training_time/60:.1f} min)")
        
        # ===== ÉVALUATION (À L'INTÉRIEUR du MLflow context) =====
        print("\n📊 ÉVALUATION SUR TEST SET")
        print("="*80)
        
        # Évaluation
        test_results = trainer.evaluate(test_dataset)
        
        # Log métriques de test dans MLflow
        mlflow.log_metric("test_accuracy", test_results['eval_accuracy'])
        mlflow.log_metric("test_precision", test_results['eval_precision'])
        mlflow.log_metric("test_recall", test_results['eval_recall'])
        mlflow.log_metric("test_f1_score", test_results['eval_f1'])
        
        print(f"\n📈 Résultats TunBERT:")
        print(f"   Accuracy:  {test_results['eval_accuracy']:.4f}")
        print(f"   Precision: {test_results['eval_precision']:.4f}")
        print(f"   Recall:    {test_results['eval_recall']:.4f}")
        print(f"   F1-Score:  {test_results['eval_f1']:.4f}")
        
        # Prédictions détaillées
        predictions = trainer.predict(test_dataset)
        pred_logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        preds = pred_logits.argmax(-1)
        labels = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"\n📊 Matrice de confusion:")
        print(cm)
        
        # Sauvegarder confusion matrix
        cm_df = pd.DataFrame(cm, index=['Actual_0', 'Actual_1'], columns=['Pred_0', 'Pred_1'])
        cm_path = MODELS_DIR / 'cm_tunbert.csv'
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))
        
        # Classification report
        report = classification_report(labels, preds, target_names=['Classe 0', 'Classe 1'], digits=4)
        print(f"\n📋 Classification Report:")
        print(report)
        
        # Sauvegarder report
        report_path = MODELS_DIR / 'classification_report_tunbert.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))
        
        # ===== SAUVEGARDE DU MODÈLE =====
        print("\n💾 Sauvegarde du modèle TunBERT")
        print("-" * 80)
        
        # Sauvegarder localement
        model.save_pretrained(TUNBERT_DIR)
        tokenizer.save_pretrained(TUNBERT_DIR)
        
        print(f"✅ Modèle sauvegardé: {TUNBERT_DIR}")
        
        # Log dans MLflow
        try:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model,
                    "tokenizer": tokenizer
                },
                artifact_path="tunbert_model",
                registered_model_name="Election_TunBERT"
            )
            print("✅ Modèle loggé dans MLflow Model Registry")
        except Exception as e:
            print(f"⚠️  MLflow model logging warning: {e}")
            # Fallback: logger comme artefact
            import shutil
            shutil.make_archive(str(TUNBERT_DIR), 'zip', str(TUNBERT_DIR))
            mlflow.log_artifact(f"{TUNBERT_DIR}.zip")
        
        print()
        
        return trainer, test_results, training_time, cm

# Les fonctions evaluate_tunbert et save_tunbert_model ont été intégrées
# directement dans train_tunbert() pour que tout soit dans le contexte MLflow

# ============================================================================
# Main
# ============================================================================

def main():
    """Pipeline principal d'entraînement TunBERT"""
    try:
        # Setup
        setup_mlflow()
        device = setup_device()
        
        # Charger données
        comments, labels = load_cleaned_texts()
        train_dataset, val_dataset, test_dataset = create_datasets(comments, labels)
        
        # Charger modèle
        model, tokenizer = load_tunbert_model(device)
        
        # Tokenization
        train_dataset, val_dataset, test_dataset = tokenize_datasets(
            train_dataset, val_dataset, test_dataset, tokenizer
        )
        
        # Entraînement + Évaluation + Sauvegarde (tout dans MLflow context)
        trainer, test_results, training_time, cm = train_tunbert(
            model, tokenizer, train_dataset, val_dataset, test_dataset, device,
            epochs=3, batch_size=8, learning_rate=2e-5
        )
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TUNBERT TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"📁 Modèle: {TUNBERT_DIR}")
        print(f"🔗 MLflow: https://dagshub.com/rahmmaakhlefa/tunsent-mlops/experiments")
        print(f"\n🏆 Résultats Finaux:")
        print(f"   F1-Score: {test_results['eval_f1']:.4f}")
        print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"   Temps: {training_time/60:.1f} min")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
