import inspect
from funasr import AutoModel

def inspect_model():
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    
    print("Inspect generate method:")
    # The generate method might be dynamically added or on the internal model
    if hasattr(model, 'generate'):
        print(inspect.signature(model.generate))
    else:
        print("model.generate not found directly.")

    # Try to find the underlying inference method
    if hasattr(model, 'model'):
        print("\nInternal model methods:")
        print(dir(model.model))

if __name__ == "__main__":
    inspect_model()
