import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

def setup_text_model():
    model_id = "j-hartmann/emotion-english-distilroberta-base"
    # Find project root (2 levels up from scripts/utils)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_root, "app", "models", "text_emotion_model")
    
    print(f"Downloading and Converting {model_id} to ONNX...")
    print(f"Output Directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load and Export to ONNX
    # This automatically handles the conversion
    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            model_id, 
            export=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("Success! Model converted and saved.")
        print(f"ONNX Model: {os.path.join(output_dir, 'model.onnx')}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Ensure you have installed: pip install optimum[exporters] onnx onnxruntime")

if __name__ == "__main__":
    setup_text_model()
