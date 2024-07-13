from flask import Flask, render_template, request, url_for, redirect
from youtube_transcript_api import YouTubeTranscriptApi as YTapi
from app import extract_video_id, get_translation_languages, translate_video
from text_summarizer import summarize_text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)


def extract_video_id(link):
    if 'youtube.com' in link:
        video_id = link.split("=")[1]
    else:
        video_id = link.split('/')[-1].split('?')[0]
    return video_id


def get_translation_languages(youtube_link):
    video_id = extract_video_id(youtube_link)
    transcript_list = YTapi.list_transcripts(video_id)

    languages = []
    for transcript in transcript_list:
        if transcript.is_translatable:
            translations = [(lang['language'], lang['language_code']) for lang in transcript.translation_languages]
            languages.append((transcript.language, translations))

    return languages


def translate_video(youtube_link, target_language):
    video_id = extract_video_id(youtube_link)
    transcript_list = YTapi.list_transcripts(video_id)

    for transcript in transcript_list:
        if transcript.is_translatable and any(
                language['language_code'] == target_language for language in transcript.translation_languages):
            translated_transcript = transcript.translate(target_language).fetch()
            translated_text = ' '.join([segment['text'] for segment in translated_transcript])
            return translated_text
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/languages', methods=['POST'])
def languages():
    youtube_link = request.form['youtube_link']
    languages = get_translation_languages(youtube_link)
    return render_template('languages.html', languages=languages)


@app.route('/translate', methods=['POST'])
def translate():
    youtube_link = request.form['youtube_link']
    target_language = request.form['target_language']
    translated_text = translate_video(youtube_link, target_language)
    return render_template('result.html', translated_text=translated_text)


@app.route('/abstract')
def abstract():
    youtube_link = request.args.get('youtube_link')
    video_id = extract_video_id(youtube_link)
    transcript_list = YTapi.list_transcripts(video_id)

    # Assume the target language is specified in the URL query parameter 'target_language'
    target_language = request.args.get('target_language', 'en')  # Default to English if not specified
    translated_text = None

    for transcript in transcript_list:
        if transcript.is_translatable and any(
                language['language_code'] == target_language for language in transcript.translation_languages):
            translated_transcript = transcript.translate(target_language).fetch()
            translated_text = ' '.join([segment['text'] for segment in translated_transcript])
            break

    if translated_text:
        summarized_text = summarize_text(translated_text)
        summarized_text_str = ' '.join(summarized_text)
    else:
        summarized_text_str = "Translation not available."

    return render_template('abstract.html', summarized_text=summarized_text_str, youtube_link=youtube_link)


@app.route('/question_answer.html')
def show_question_answer():
    return render_template('question_answer.html')


# Route for asking questions
@app.route('/question_answer', methods=['GET', 'POST'])
def question_answer():
    if request.method == 'POST':
        # Get the user question from the form
        user_question = request.form['question']

        # Generate answer using the provided function
        answer = generate_answer(user_question)

        # Render the question and answer template with the generated answer
        return render_template('question_answer.html', question=user_question, answer=answer)
    else:
        # Render the question form template
        return render_template('ask_question.html')


# Function to generate answer to user question
def generate_answer(question):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Load model architecture
    model_checkpoint = "gpt2"  # Example checkpoint, change it to your model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Load the model parameters
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Load on CPU

    # Tokenize the question
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate answer
    output = model.generate(inputs, max_length=50, num_return_sequences=1, early_stopping=True)

    # Decode and return the answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


@app.route('/ask_question')
def ask_question():
    return render_template('ask_question.html')


@app.route('/back_to_ask_question')
def back_to_ask_question():
    # Redirect to the ask_question route
    return redirect(url_for('ask_question'))


if __name__ == '__main__':
    app.run(debug=True)
