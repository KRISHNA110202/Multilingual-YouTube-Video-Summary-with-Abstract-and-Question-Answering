from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi as YTapi

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


if __name__ == '__main__':
    app.run(debug=True)
