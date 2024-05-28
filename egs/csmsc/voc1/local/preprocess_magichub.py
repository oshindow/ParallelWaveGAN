from pydub import AudioSegment
import re
import os

def is_chinese_char(char):
    # Check if a character is a Chinese character
    return '\u4e00' <= char <= '\u9fff'

def contains_only_chinese(text):
    # Check if a text contains only Chinese characters
    return all(is_chinese_char(c) or c in '，。？！：；“”‘’《》〈〉【】[]()（）·' for c in text)

def process_file(input_text_file, input_audio_file, output_folder):
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    basename = os.path.basename(input_audio_file)[:-4]
    with open(input_text_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    audio = AudioSegment.from_wav(input_audio_file)
    segment_count = 0

    for line in lines:
        parts = line.strip().split('\t')
        times = parts[0].strip('[]').split(',')
        keep_flag = int(parts[1])
        text = parts[3]

        if keep_flag == 1 and contains_only_chinese(text):
            start_time = float(times[0]) * 1000  # Convert to milliseconds
            end_time = float(times[1]) * 1000  # Convert to milliseconds
            segment = audio[start_time:end_time]
            segment_count += 1
            segment_filename = os.path.join(output_folder, f"{basename}_segment_{segment_count:04d}.wav")
            segment.export(segment_filename, format="wav")

# Usage
import os
if __name__ == '__main__':
    
    for file in os.listdir('/data2/xintong/magichub_sg/WAV'):
        input_text_file = '/data2/xintong/magichub_sg/TXT/' + file[:-4] + '.txt'  # Your input text file path
        input_audio_file = '/data2/xintong/magichub_sg/WAV/' + file # Your input audio file path
        spk = input_audio_file.split('_')[-1][:-4]
        
        output_folder = '/data2/xintong/magichub_sg/segments/' + spk  # Folder to save the segments

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        process_file(input_text_file, input_audio_file, output_folder)
