import json
from typing import List
import numpy as np
import onnxruntime
from enum import Enum
from piper_phonemize import phonemize_codepoints, phonemize_espeak
import numpy as np
import subprocess
import os
from util import encoding

PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence

class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    TEXT = "text"

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess_options = onnxruntime.SessionOptions()
        self.providers = ['CPUExecutionProvider']
        self.model, self.config = self.load()

    def audio_float_to_int16(self, audio: np.ndarray, max_wav_value: float = 32767.0) -> np.ndarray:
        audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    # def audio_array_to_opus(self,audio_array, model_sample_rate=22050, opus_sample_rate=48000):
    #     # Save the raw audio to a temporary file
    #     raw_audio_file = "temp_audio.raw"
    #     opus_audio_file = "temp_audio.opus"

    #     audio_array = audio_array.astype(np.int16)  # Ensure the audio is in int16 format
    #     with open(raw_audio_file, "wb") as raw_file:
    #         raw_file.write(audio_array.tobytes())

    #     # Use ffmpeg to convert raw PCM audio to Opus format
    #     subprocess.run([
    #         "ffmpeg", "-y",
    #         "-f", "s16le",
    #         "-ar", str(model_sample_rate),
    #         "-ac", "1",
    #         "-i", raw_audio_file,
    #         "-c:a", "libopus",
    #         "-b:a", "64k",  # Example bitrate
    #         "-ar", str(opus_sample_rate),
    #         opus_audio_file
    #     ], check=True)

    #     # Read the Opus-encoded audio as bytes
    #     with open(opus_audio_file, "rb") as opus_file:
    #         opus_audio_bytes = opus_file.read()

    #     # Clean up temporary files
    #     subprocess.run(["rm", raw_audio_file, opus_audio_file])

    #     return opus_audio_bytes

    def phonemize(self, config, text: str) -> List[List[str]]:
        phoneme_type = config.get("phoneme_type", PhonemeType.ESPEAK.value)
        try:
            if phoneme_type == PhonemeType.ESPEAK:
                return phonemize_espeak(text, config["espeak"]["voice"])
            if phoneme_type == PhonemeType.TEXT:
                return phonemize_codepoints(text)

        # try:
        #     if config["phoneme_type"] == PhonemeType.ESPEAK:
        #         return phonemize_espeak(text, config["espeak"]["voice"])
        #     if config["phoneme_type"] == PhonemeType.TEXT:
        #         return phonemize_codepoints(text)
        except Exception as e:
            print("PHONEMIZER:",e)
            raise ValueError(f'Unexpected phoneme type: {config["phoneme_type"]}')

    def phonemes_to_ids(self, config, phonemes: List[str]) -> List[int]:
        id_map = config["phoneme_id_map"]
        ids: List[int] = list(id_map[BOS])
        for phoneme in phonemes:
            if phoneme not in id_map:
                print(f"Missing phoneme from id map: {phoneme}")
                continue
            ids.extend(id_map[phoneme])
            ids.extend(id_map[PAD])
        ids.extend(id_map[EOS])
        return ids

    def load(self):
        print(f"LOADING MODEL {self.model_path}")
        config_path = f"{self.model_path}.json"
        with open(config_path, "r") as file:
            config = json.load(file)

        model = onnxruntime.InferenceSession(
            str(self.model_path),
            sess_options=self.sess_options,
            providers=self.providers
        )
        return model,config

    def inference(self, model, config, sid, line, length_scale, noise_scale, noise_scale_w):
        print("MODEL", model)
        try:
            audios = []
            num_speakers = config["num_speakers"]
            speaker_id = None if num_speakers ==1 else sid
            
            scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)
            #print(scales)
            sid = None
            if speaker_id is not None:
                sid = np.array([speaker_id], dtype=np.int64)
            sentence_silence = 0.4
            phonemized_text = self.phonemize(config, line)
            num_silence_samples = int(sentence_silence * 22050)
            silence_bytes = bytes(num_silence_samples * 2)
            
            #print("PT", phonemized_text)
            for phonemes in phonemized_text:
                phoneme_ids = self.phonemes_to_ids(config, phonemes)
                #print("Phoneme ids" ,phoneme_ids)
                expanded_text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
                #print("expanded_text" ,expanded_text)
                text_lengths = np.array([expanded_text.shape[1]], dtype=np.int64)
                #print("text lengths", text_lengths)
                
                
                #print(f"IDS {phoneme_ids}\n\n TEXT{expanded_text}")
                
                #audio = model.run(None,{"input": expanded_text,"input_lengths": text_lengths,"scales": scales,"sid": sid,},)[0].squeeze((0,1))
                audio = model.run(
                None,
                {
                    "input": expanded_text,
                    "input_lengths": text_lengths,
                    "scales": scales,
                    "sid": sid,
                },
                )[0].squeeze((0,1))
                
                #print(f"Audio shape: {audio.shape}")
                audio = self.audio_float_to_int16(audio.squeeze())
                audios.append(audio)
                #print("before" , audios)
                audios.append(np.frombuffer(silence_bytes, dtype=np.int16))
                #print("after" , audios)

            merged_audio = np.concatenate(audios)
            return merged_audio
    
        except Exception as e:
            print( "EXCEPTION", e)

    def run_inference(self, text, speaker_id, **kwargs):
        text = text.lower().replace('.' , '.>>')
        length_scale_test = kwargs.get("length_scale" , 1.3)
        print("LST" , length_scale_test)
        #print(kwargs)
        audio_array = self.inference(
            model = self.model,
            config=self.config,
            sid=speaker_id,
            line=text,
            #length_scale=kwargs.get("length_scale", 1.45),
            length_scale = length_scale_test,
            noise_scale=kwargs.get("noise_scale", 0.33),
            noise_scale_w=kwargs.get("noise_scale_w", 0.33),
        )
        
        print("AUDIO ARRAY", audio_array)
        try:
            #encoded_audio = encoding.resample_and_encode(audio_array , 22050, kwargs.get('sample_rate_hertz'), kwargs.get('audio_encoding'))
            encoded_audio = encoding.resample_and_encode(audio_array , 22050, 48000, 'OGG_OPUS')
        except Exception as e:
            print("encode error", e)
        
        return encoded_audio