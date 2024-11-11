import time
from pygame import mixer
import os
import pyttsx3  # Local TTS library
from transformers import AutoTokenizer, AutoModelForCausalLM  # For local model

# Initialize the local model and tokenizer
model_name = "path/to/your/local/model"  # Replace with the actual path to your local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the mixer for audio playback
mixer.init()

# Initialize pyttsx3 TTS engine for local TTS
tts_engine = pyttsx3.init()

# Function to use the local model for generating responses
def ask_question_memory(question):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt")
    # Generate a response using the model
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
    # Decode and return the response text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate TTS locally and save it as a file
def generate_tts(sentence, speech_file_path):
    tts_engine.save_to_file(sentence, speech_file_path)
    tts_engine.runAndWait()
    return speech_file_path

# Play the generated TTS audio
def play_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

# Main TTS function that generates audio and plays it
def tts(text):
    speech_file_path = generate_tts(text, "speech.mp3")
    play_sound(speech_file_path)
    while mixer.music.get_busy():
        time.sleep(1)
    mixer.music.unload()
    os.remove(speech_file_path)
    return "done"

# Example call (uncomment to test in isolation)
# question = "What is the weather today?"
# response = ask_question_memory(question)
# print(response)
