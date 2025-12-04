import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# ================================
# CPU-COMPATIBLE BASE MODEL (Important!)
# ================================
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
LORA_ADAPTER = "mihailocvetkovic/SML-Lab-2-Model"

device = "cpu"

print("\n📌 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("\n📌 Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,        # CPU-safe dtype
    device_map={"": device},          # Force full CPU load
)

print("\n📌 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model = model.to(device)
model.eval()

# ================================
# RESPONSE GENERATION
# ================================
def generate_response(user_input, history):
    try:
        prompt = f"<|begin_of_text|>User: {user_input}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean formatting if needed
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[-1].strip()

        return response

    except Exception as e:
        return f"❌ ERROR: {str(e)}"


# ================================
# GRADIO UI
# ================================
chatbot = gr.ChatInterface(
    fn=generate_response,
    title="SML-Lab-2 Fine-Tuned Chatbot (CPU Optimized)",
)

chatbot.launch()






