import random
from models.text_detection import is_generated_by_ai  
from flask import Flask, request, render_template, redirect, url_for
import os
from models.deepfake_detection import load_models, predict

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ensemble_model = load_models('../CelebDF_model_20_epochs_99acc.pt', '../FFPP_model_20_epochs_99acc.pt')

question_bank = [
    {"question": "What is a deepfake?", "options": ["A fake video or audio created using AI", "A type of malware", "A deep-sea creature", "A computer game"], "answer": "A fake video or audio created using AI"},
    {"question": "Which technology is commonly used to create deepfakes?", "options": ["Blockchain", "AI and Machine Learning", "Quantum Computing", "3D Printing"], "answer": "AI and Machine Learning"},
    {"question": "What are GANs in the context of deepfakes?", "options": ["Global Address Networks", "Generative Adversarial Networks", "Generalized Artificial Networks", "Graphical Array Networks"], "answer": "Generative Adversarial Networks"},
    {"question": "Who is commonly targeted by deepfakes?", "options": ["Politicians and celebrities", "Teachers", "Doctors", "Engineers"], "answer": "Politicians and celebrities"},
    {"question": "What is one of the main risks of deepfakes?", "options": ["Enhanced video game graphics", "Spread of misinformation", "Improved movie special effects", "Better photo editing"], "answer": "Spread of misinformation"},
    {"question": "How can deepfakes be detected?", "options": ["By listening for background noise", "Using forensic analysis and AI tools", "By checking the file size", "By analyzing the file name"], "answer": "Using forensic analysis and AI tools"},
    {"question": "What does the term 'adversarial' mean in Generative Adversarial Networks?", "options": ["Cooperative", "Competing", "Neutral", "Supportive"], "answer": "Competing"},
    {"question": "Which industry is most impacted by deepfakes?", "options": ["Healthcare", "Automotive", "Entertainment and media", "Agriculture"], "answer": "Entertainment and media"},
    {"question": "What is one way to protect against deepfakes?", "options": ["Using stronger passwords", "Implementing digital watermarking", "Upgrading hardware", "Improving internet speed"], "answer": "Implementing digital watermarking"},
    {"question": "What year did deepfakes first gain significant attention?", "options": ["2001", "2015", "2020", "1995"], "answer": "2015"},
    {"question": "What is 'face swapping' in the context of deepfakes?", "options": ["Exchanging faces in real-time", "Changing the background of an image", "Replacing one person's face with another's in videos or photos", "Adjusting facial expressions"], "answer": "Replacing one person's face with another's in videos or photos"},
    {"question": "Which of the following is a positive use of deepfake technology?", "options": ["Political manipulation", "Identity theft", "Educational content creation", "Creating fake news"], "answer": "Educational content creation"},
    {"question": "Can deepfakes be created using open-source software?", "options": ["Yes", "No"], "answer": "Yes"},
    {"question": "What is 'voice cloning'?", "options": ["Mimicking a person's voice using AI", "Recording an audiobook", "Improving voice clarity", "Changing the pitch of a voice"], "answer": "Mimicking a person's voice using AI"},
    {"question": "What is the main ethical concern with deepfakes?", "options": ["Data storage", "Privacy invasion and consent", "Network speed", "Energy consumption"], "answer": "Privacy invasion and consent"},
    {"question": "What role do convolutional neural networks (CNNs) play in deepfakes?", "options": ["Data encryption", "Image and video processing", "Network management", "File compression"], "answer": "Image and video processing"},
    {"question": "What is the potential impact of deepfakes on elections?", "options": ["No impact", "Positive impact", "Neutral impact", "Negative impact due to misinformation"], "answer": "Negative impact due to misinformation"},
    {"question": "Which platform has taken measures to combat deepfakes?", "options": ["Facebook", "LinkedIn", "GitHub", "Pinterest"], "answer": "Facebook"},
    {"question": "How does deepfake technology affect personal security?", "options": ["It enhances security", "It can be used for identity theft", "It has no effect", "It improves authentication methods"], "answer": "It can be used for identity theft"},
    {"question": "What is 'data poisoning' in the context of deepfakes?", "options": ["Adding malicious data to corrupt models", "Cleaning data", "Encrypting data", "Backing up data"], "answer": "Adding malicious data to corrupt models"},
    {"question": "Which of the following is a legal measure against deepfakes?", "options": ["Open-source licensing", "Anti-deepfake legislation", "Software patents", "Network firewalls"], "answer": "Anti-deepfake legislation"},
    {"question": "Can deepfake technology be used for positive purposes?", "options": ["Yes", "No"], "answer": "Yes"},
    {"question": "What does the discriminator do in a GAN?", "options": ["Creates fake data", "Classifies real and fake data", "Encrypts data", "Compresses data"], "answer": "Classifies real and fake data"},
    {"question": "What is the first step in creating a deepfake?", "options": ["Data collection", "Data encryption", "Data compression", "Data deletion"], "answer": "Data collection"},
    {"question": "How can individuals protect themselves from being targeted by deepfakes?", "options": ["Avoiding social media", "Regularly updating software", "Using secure passwords", "Being cautious with sharing personal media"], "answer": "Being cautious with sharing personal media"}
]


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/text-detection')
def text_detection():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    para = request.form['text']
    result = is_generated_by_ai(para)
    print(para)
    result_text = "Generated by AI" if result else "Written by a human"
    return render_template('result.html', result=result_text)

@app.route('/quiz')
def quiz():
    questions = random.sample(question_bank, 10)
    return render_template('quiz.html', questions=questions)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    total_score = 0
    for question in question_bank:
        user_answer = request.form.get(question['question'])
        if user_answer == question['answer']:
            total_score += 1
    return render_template('quiz_result.html', score=total_score)

@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("No image part in the request")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            print("No file selected for uploading")
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            print(f"File saved to {filepath}")
            result = predict(filepath, ensemble_model)
            print(f"Prediction result: {result}")
            return render_template('result_deepfake.html', result=result)
    return render_template('image_detection.html')

if __name__ == '__main__':
    app.run(debug=True)