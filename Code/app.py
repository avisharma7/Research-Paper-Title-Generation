from flask import Flask, render_template, request
from difflib import SequenceMatcher
import openai
import os
import PyPDF2
import json
from datetime import date
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Set up your OpenAI API key
openai.api_key = "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"

# Define a function to calculate the similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Define a function to calculate the accuracy of a generated title
def calculate_accuracy(generated_title, actual_title):
    return similar(generated_title, actual_title) * 100

# Define a function to read a PDF file and extract its text
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for i in range(len(reader.pages)):
            # Get the i-th page and extract its text
            page = reader.pages[i]
            page_text = page.extract_text()

            # Add the extracted text to the final result
            text += page_text
        return text

# Define a function to generate a title for a given PDF file
def generate_title_from_pdf(pdf_path, actual_title=None):
    # Extract the text from the PDF file
    text = extract_text_from_pdf(pdf_path)

    # Set up the OpenAI API request
    prompt = f"Please generate a title for the following paper: \n\n{text}\n\nTitle:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated title from the API response
    title = response.choices[0].text.strip()

    if actual_title:
        accuracy = calculate_accuracy(title, actual_title)
    else:
        accuracy = None

    return title, accuracy

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to the local file system
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Generate the title for the file
        actual_title = request.form.get('actual_title')
        title,accuracy = generate_title_from_pdf(file_path, actual_title)
        
        #store accuracy in json file
        with open('data.json', 'r') as f:
            data = json.load(f)

        # Append new data to the dictionary
            new_data = {'accuracy': accuracy,'id':len(data)+1}
            data.append(new_data)

        # Write the updated dictionary back to the JSON file
        with open('data.json', 'w') as f:
            json.dump(data, f)

        # create a chart
        # Extract the accuracy values
        accuracies = [d['accuracy'] for d in data]

        # Plot a line chart of the accuracy values
        fig, ax = plt.subplots()
        ax.plot(accuracies)
        ax.set_xlabel('Id')
        ax.set_ylabel('Accuracy')

        # Convert the plot to a base64-encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        # Render the result page with the generated title
        return render_template('result.html', title=title, accuracy=accuracy, plot_data=plot_data)
    
    else:
        # Render the upload form page
        return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
