from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re,os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from box import Box
import yaml

with open("config.yaml") as f:
    cfg = Box(yaml.safe_load(f))
root = cfg.dataset.path_to_data

# Download necessary NLTK data
nltk.download('punkt')


def extract_text_from_html(html_file):

    # Open the HTML file
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Extract the title
    title = soup.find('h1', {'id': 'firstHeading'}).get_text()

    # Extract text from the 'mw-parser-output' class
    content = soup.find('div', {'class': 'mw-parser-output'})
    paragraphs = content.find_all('p')

    # Save the extracted title and text to a new file
    with open("tmp.txt", "w", encoding="utf-8") as output_file:
        for paragraph in paragraphs:
            output_file.write(paragraph.get_text() + "\n")

def cleaning_text(n, category):
    with open("tmp.txt", "r", encoding="utf-8") as file:
        text = file.readlines()

    # Join the lines into a single string
    text = ''.join(text)

    # Remove square brackets and any numeric characters enclosed within them
    # Also remove [Note ... ]
    cleaned_text = re.sub(r'\[[^\]]*\]', '', text)

    # Now tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)
    # Tokenize sentences into words and maintain a list of token sequences
    sequences = []
    current_sequence = []

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if len(current_sequence) + len(tokens) <= n:
            current_sequence.extend(tokens)
        else:
            sequences.append(current_sequence)
            current_sequence = tokens
            if len(current_sequence) > n:
                sequences.append(current_sequence[:n])
                current_sequence = current_sequence[n:]

    if current_sequence:
        sequences.append(current_sequence)

    # Convert token sequences back to text sequences
    text_sequences = [" ".join(seq) for seq in sequences]

    # Print or save the sequences
    for i, seq in enumerate(text_sequences):
        # Optionally, save each sequence to a separate file
        with open(f"{root}/text_dataset/sequences/{category}/sequence_{i}.txt", "w", encoding="utf-8") as file:
            file.write(seq)

def generate_questions(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    input_text = "Your text sequence goes here."
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, num_return_sequences=5, temperature=0.7)   
    for i in range(5):  # Assuming we asked for 5 sequences
        question = tokenizer.decode(output[i], skip_special_tokens=True)
        print(question) 


if __name__ == "__main__":
    extract_text_from_html("dataset/cataract_text_dataset/htmls/Manual_small_incision_cataract_surgery.html")
    cleaning_text(n=300, category="incision")
    os.system("rm tmp.txt")

    text = open("sequence_1.txt","r")
    print(text)
    generate_questions(text)