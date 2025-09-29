import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.io import wavfile
import librosa
import os
import csv

model = hub.load('https://tfhub.dev/google/yamnet/1')


def load_and_preprocess_wav(file_path, desired_sample_rate=16000):
    try:
        sample_rate, wav_data = wavfile.read(file_path)

        if wav_data.dtype == np.float32 or wav_data.dtype == np.float64:
            waveform = wav_data.astype(np.float32)
        else:
            waveform = wav_data.astype(
                np.float32) / np.iinfo(wav_data.dtype).max

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != desired_sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=desired_sample_rate)
            sample_rate = desired_sample_rate

        return waveform, sample_rate

    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        try:
            waveform, sample_rate = librosa.load(
                file_path, sr=desired_sample_rate, mono=True)
            return waveform, sample_rate
        except Exception as e2:
            print(f"Ошибка и при альтернативной загрузке: {e2}")
            return None, None


class_map_path = model.class_map_path().numpy()


def load_class_names(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names


class_names = load_class_names(class_map_path)

for file in os.listdir("sound/"):
    waveform, sample_rate = load_and_preprocess_wav(f"sound/{file}")
    print("\n\n")
    if waveform is not None:
        print(f"Аудио загружено: {len(waveform)} samples, {sample_rate} Hz")

        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()

        mean_scores = np.mean(scores_np, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]

        print(f"\nРезультаты для файла {file}:")
        for i, idx in enumerate(top_indices):
            class_name = class_names[idx] if idx < len(
                class_names) else f"Class_{idx}"
            confidence = mean_scores[idx]
            print(f"{i+1}. {class_name}: {confidence:.4f}")
    else:
        print("Не удалось загрузить аудиофайл")
