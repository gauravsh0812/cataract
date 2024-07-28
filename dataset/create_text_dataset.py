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

def get_data_from_website(category):
    for i,url in enumerate(open(f"{root}/text_dataset/links/{category}_links.lst").readlines()):
        cmd = f"wget -O {root}/text_dataset/htmls/link_{i}.html '{url}'"
        os.system(cmd)

def extract_text_from_html(html_file):

        html_file_path = f"{root}/text_dataset/htmls/{html_file}"
        
        # Open the HTML file
        with open(html_file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Attempt to find the title
        title_element = soup.find('h1', {'id': 'firstHeading'})

        if title_element:
            title = title_element.get_text()
        else:
            print(f"No title found in {html_file}")
            title = "No Title Found"

        # Extract text from the 'mw-parser-output' class
        content = soup.find('div', {'class': 'mw-parser-output'})
        if content:
            paragraphs = content.find_all('p')
        else:
            print(f"Content not found in {html_file}")
            paragraphs = []

        # Save the extracted title and text to a new file
        with open("tmp.txt", "w", encoding="utf-8") as output_file:
            for paragraph in paragraphs:
                output_file.write(paragraph.get_text() + "\n")

def cleaning_text(n, category, count):

    os.makedirs(f"{root}/text_dataset/sequences/{category}", exist_ok=True)

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
        with open(f"{root}/text_dataset/sequences/{category}/sequence_{count+i}.txt", "w", encoding="utf-8") as file:
            file.write(seq)
        count+=1
    
    return count

def generate_questions(text):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_question(text_sequence):
    # Load pre-trained model and tokenizer
    model_name = 'gpt2'  # You can use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger models
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Define the prompt to generate the question
    prompt = f"Given the following text sequence:\n\n{text_sequence}\n\nGenerate a question that can be associated with this sequence:\n\nQ:"

    # Encode the input and generate output
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1)

    # Decode the generated question
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract and return the generated question
    question_start = question.find("Q:") + 2
    generated_question = question[question_start:].strip()
    return generated_question



if __name__ == "__main__":
    category = "incision"
    get_data_from_website(category)

    count = 0
    for html_file in os.listdir(f"{root}/text_dataset/htmls"):
        extract_text_from_html(html_file)
        count = cleaning_text(n=100, category=category, count=count)
        os.system("rm tmp.txt")


    for _f in os.listdir(f"{root}/text_dataset/sequences/{category}"):
        text = open(f"{root}/text_dataset/sequences/{category}/{_f}","r")
        print(text)
        # Generate and print the question
        generated_question = generate_question(text)
        print(f"Generated Question: {generated_question}")
