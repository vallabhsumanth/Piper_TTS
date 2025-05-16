import grpc
from protos import piper_pb2
from protos import piper_pb2_grpc

def get_language():
    channel = grpc.insecure_channel('localhost:50052')
    stub = piper_pb2_grpc.gwStub(channel)

    # Get Languages
    languages_response = stub.GetLanguages(piper_pb2.GetLanguagesRequest())
    print("Available Languages:")
    for language in languages_response.languages:
        print(f"Language: {language.language},  ISO: {language.iso_code}, Language ID: {language.language_id}")
    
    voices_response = stub.GetVoices(piper_pb2.GetVoicesRequest(language_id="en-US"))
    for voice in voices_response.voices:
        print(f"Voice ID: {voice.voice_id}, Name: {voice.voice_name}")

    # # Assuming you choose the first language and get voices for it (if available)
    # if languages_response.languages:
    #     language_id = languages_response.languages[0].language_id
    #     voices_response = stub.GetVoices(tts_pb2.GetVoicesRequest(language_id=language_id))
    #     print("\nAvailable Voices for the selected language:")
    #     for voice in voices_response.voices:
    #         print(f"Voice ID: {voice.voice_id}, Name: {voice.voice_name}")

def synthesize_and_save(request_id):
    # Assuming the server is running on localhost at port 50058
    channel = grpc.insecure_channel('localhost:50052')
    stub = piper_pb2_grpc.gwStub(channel)

    # Example request
    #text = "Hello.Please listen to the inputs carefully as they have changed.Press 1 if you have a query related to your account.Press 2 if you are having issues accessing the portal. Press 3 if you want to speak to a representative"
    #text = "Hello ?? You can call me at 9 7 0 2 7 3 2 0 7 2"
    #text = "Kindly stay on the line. You are 23rd in queue. Your wait time is 7 minutes"
    text = "Welcome to Barclays contact center solution services.Listen carefully as our options have changed.Press 1 for leaving this call.Press 2 for creating a new ticket.Press 3 for talking to an agent immediately"
    #text = "Hello"
    # voice_name = "en_US_ljspeech"  # Use the correct voice name for the request
    voice_id = "en-US-01"
    language_code = "en-US"  # Example language code
    sample_rate_hertz = 48000  # Desired Opus sample rate
    audio_encoding = piper_pb2.OGG_OPUS  # Set encoding to Opus

    # Create the request
    request = piper_pb2.SynthesizeSpeechRequest(
        input_text=text,
        voice=piper_pb2.VoiceSelectionParams(
            language_code=language_code,
            voice_id=voice_id
        ),
        audio_config=piper_pb2.AudioConfig(
            audio_encoding=audio_encoding,
            sample_rate_hertz=sample_rate_hertz
        )
    )

    opus_audio_file = f"./audio_{request_id}.opus"
    
    # Stream and save Opus audio chunks
    with open(opus_audio_file, "wb") as opus_file:
        try:
            for response in stub.Synthesize(request):
                #print("AUDIO CONTENT", response.audio_content)
                opus_file.write(response.audio_content)
        except grpc.RpcError as e:
            print(f"RPC failed: {e.code()}, {e.details()}")
            return None

    print(f"Opus audio saved at: {opus_audio_file}")
    return opus_audio_file

# Example usage
if __name__ == "__main__":
    output_file = synthesize_and_save(request_id=3)
    # if output_file:
    #     print(f"Audio saved at: {output_file}")
    # get_language()
    
## circle [normal call] - gw - 
## qdrant search 