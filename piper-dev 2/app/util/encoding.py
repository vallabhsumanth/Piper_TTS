import numpy as np
import io
import subprocess


def resample_and_encode(audio_array, original_sr, target_sr, audio_encoding):
    # Convert the numpy array to a byte stream
    audio_array = np.asarray(audio_array, dtype=np.int16)
    audio_stream = io.BytesIO(audio_array.tobytes())

    # Determine the ffmpeg command based on the encoding
    if audio_encoding == 'LINEAR16':
        output_format = 's16le'
    elif audio_encoding == 'MULAW':
        output_format = 'mulaw'
    elif audio_encoding == 'MP3':
        output_format = 'mp3'
    elif audio_encoding == 'OGG_OPUS':
        output_format = 'opus'
    else:
        raise ValueError("Unsupported audio encoding")

    # Construct the ffmpeg command
    # command = [
    #     'ffmpeg',
    #     '-f', 's16le',  # Input format is signed 16-bit little-endian
    #     '-ar', str(original_sr),  # Input sample rate
    #     '-i', 'pipe:',  # Input from pipe
    #     '-ar', str(target_sr),  # Output sample rate
    #     '-ac', '1',  # Mono audio
    #     '-f', output_format,  # Output format
    #     'pipe:'  # Output to pipe
    # ]
    
    command = [
    'ffmpeg',
    '-f', 's16le',  
    '-ar', str(original_sr),  
    '-i', 'pipe:', 
    '-ar', str(target_sr),
    '-ac', '1',
    '-af', 'equalizer=f=100:t=q:w=2:g=10,loudnorm',  # Add bass and normalize audio
    '-f', output_format,
    'pipe:'
    ]

    # Run the ffmpeg command
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = process.communicate(input=audio_stream.read())
    
    #print("OUTPUT OF ENCODING", output)
    return output