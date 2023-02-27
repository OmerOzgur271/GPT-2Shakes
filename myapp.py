# -*- coding: utf-8 -*-




import wget


from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from fastapi import FastAPI
import os
import uvicorn
import requests
import threading

url = 'https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt'
filename = wget.download(url)
# Create a tokenizer object based on the GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create a TextDataset object that represents the input data for the model
# The dataset is initialized with the tokenizer, the file path of the input text file, and a block size of 128
dataset = TextDataset(
    tokenizer=tokenizer,     # Tokenizer object
    file_path='input.txt',   # Path of input text file
    block_size=128           # Maximum length of input sequences
)

# Create a DataCollatorForLanguageModeling object
# This is used to collate the input data into batches for the model
# We set the mlm (masked language modeling) flag to False, since we are not using it here
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,   # Tokenizer object
    mlm=False              # Set the mlm flag to False
)

# Create a GPT2LMHeadModel object based on the pre-trained 'gpt2' model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize the token embeddings of the model to match the size of the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Set up the training arguments for the model
# We set the output directory, the number of training epochs, the batch size, and the save steps and total limit for saving models
training_args = TrainingArguments(
    output_dir='./results',                    # Output directory for saving models
    overwrite_output_dir=True,                 # Overwrite the output directory if it exists
    num_train_epochs=1,                        # Number of training epochs
    per_device_train_batch_size=16,            # Batch size for training
    save_steps=1000,                           # Save the model after every 1000 steps
    save_total_limit=2                         # Limit the total number of saved models to 2
)

# Create a Trainer object for training the model
# We set the model, training arguments, dataset, and data collator as arguments
trainer = Trainer(
    model=model,                               # GPT2LMHeadModel object
    args=training_args,                        # TrainingArguments object
    train_dataset=dataset,                     # TextDataset object
    data_collator=data_collator                 # DataCollatorForLanguageModeling object
)

print('Training Starting!!!!')

# Train the model
trainer.train()

# Save the trained model to a file
trainer.save_model('shakespeare_gpt2')

# Save the tokenizer object to a file
tokenizer.save_pretrained('shakespeare_gpt2')

# Load the pre-trained tokenizer and model for Shakespeare
tokenizer = GPT2Tokenizer.from_pretrained('shakespeare_gpt2')
model = GPT2LMHeadModel.from_pretrained('shakespeare_gpt2')

# Create a FastAPI instance
app = FastAPI()

# Define a route for generating text
@app.get('/generate_text/')
def generate_text_get(prompt: str, length: int = 50, temperature: float = 1.0):
    # Encode the input prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text using the model
    output = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output tokens into text and return it
    return {'text': tokenizer.decode(output[0], skip_special_tokens=True)}

# Define a function to run the server
def run_server():
    # Start the server using uvicorn
    uvicorn.run(app, port=8000, host='0.0.0.0')

# Start the server in a separate thread
if __name__ == '__main__':
    # Create a new thread for the server
    server_thread = threading.Thread(target=run_server)
    # Start the thread
    server_thread.start()

