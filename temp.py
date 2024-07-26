
from transformers import pipeline

# Load a pre-trained question generation model
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")

context = """
Cataract surgery is a procedure to remove the lens of your eye and, in most cases, replace it with an artificial lens. Cataracts cause the lens to become cloudy, which impairs vision. The surgery is usually performed on an outpatient basis and takes about an hour. It is a common and generally safe procedure, with a high success rate in improving vision.
"""

# Preprocess the context for question generation
context_with_highlights = "generate questions: " + context.replace(". ", " </s> ")

# Generate questions
questions = question_generator(context_with_highlights, max_length=512, do_sample=False)

# Print the generated questions
for i, question in enumerate(questions):
    print(f"Q{i+1}: {question['generated_text']}")

