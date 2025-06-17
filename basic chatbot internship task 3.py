import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None  # to maintain conversation context

# GUI send function
def send():
    global chat_history_ids

    user_input = entry_box.get()
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    entry_box.delete(0, tk.END)

    if user_input.lower() == "bye":
        chat_log.insert(tk.END, "Bot: Goodbye!\n")
        return

    # Encode user input and add to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    chat_log.insert(tk.END, "Bot: " + response + "\n")

# GUI Setup
root = tk.Tk()
root.title("AI Chatbot (DialoGPT)")

chat_log = tk.Text(root, bg="lightgray", width=60, height=20)
chat_log.pack(padx=10, pady=10)

entry_box = tk.Entry(root, width=50)
entry_box.pack(padx=10, pady=(0,10))
entry_box.bind("<Return>", lambda event: send())

send_button = tk.Button(root, text="Send", width=10, command=send)
send_button.pack()

chat_log.insert(tk.END, "Bot: Hello! I'm an AI chatbot. Ask me anything.\n")

root.mainloop()
