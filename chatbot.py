import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

try:
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
    tokenizer.save_pretrained('./gpt2_mental_health_1')

    model = GPT2LMHeadModel.from_pretrained('gpt2')  
    model.save_pretrained('./gpt2_mental_health_1')

    model.eval()  # Set model to evaluation mode

    # Set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model loaded successfully on {device}")

    # Set pad_token to eos_token to avoid padding issues
    tokenizer.pad_token = tokenizer.eos_token

except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

def clean_text(text):
    """Clean and preprocess text as done during training"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

def generate_response(user_input):
    """Generate chatbot response"""
    if not model or not tokenizer:
        return "Sorry, the model is not available."

    try:
        # Clean the input
        cleaned_input = clean_text(user_input)
        input_text = cleaned_input + " [SEP]"

        # Tokenize the input
        encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_length=150,
                min_length=20,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )

        # Decode the response
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Clean up the response
        response = response.strip()
        if not response:
            response = "I'm here to listen. Could you tell me more about what you're feeling?"

        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble understanding. Could you please rephrase?"

def chat():
    """Start a chat session"""
    print("Chatbot: Hi! I'm here to talk. How can I help you today?")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Take care.")
            break

        # Generate response
        bot_response = generate_response(user_message)

        print(f"Chatbot: {bot_response}")

if __name__ == '__main__':
    chat()
