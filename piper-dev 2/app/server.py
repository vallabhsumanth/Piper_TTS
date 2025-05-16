import grpc
from concurrent import futures
from protos import piper_pb2
from protos import piper_pb2_grpc
from model import Model
import os
import yaml
from google.protobuf.json_format import MessageToDict

# Load configuration from YAML file
CONFIG_FILE = os.getenv('CONFIG_PATH', "/workspace/piper/piper/app/models.yaml")
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

# Initialize models
models = {}
for model_config in config["models"]:
    model_name = model_config["name"]
    model_path = model_config["path"]
    models[model_name] = Model(model_path)


# Map voices to model and speaker details
voices = {voice["voice_id"]: voice for voice in config["voices"]}
print("Loaded voices:", voices)
class gwSpeechServicer(piper_pb2_grpc.gwServicer):
    def GetLanguages(self, request, context):
        """Returns a list of supported languages."""
        languages = config.get("languages", [])
        print("Supported languages:", languages)
        response = piper_pb2.GetLanguagesResponse()
        for lang in languages:
            print(f"Adding language: {lang['name']} with iso_code: {lang['iso_code']}, language_id: {lang['language_id']}")
            response.languages.add(language=lang["name"], iso_code=lang["iso_code"], language_id=lang["language_id"])
        return response

    def GetVoices(self, request, context):
        print("Received request for voices")
        """Returns a list of voices for the given language."""
        voices_list = config["voices"]
        response = piper_pb2.GetVoicesResponse()
        for voice in voices_list:
            print(f"Checking voice: {voice['name']} for language {request.language_id}")
            if voice["language"] == request.language_id:
                print(f"Adding voice: {voice['name']} for language {request.language_id}")
                response.voices.add(
                    voice_id=voice["voice_id"],
                    voice_name=voice["name"]
                )
        return response

    def Synthesize(self, request, context):
        print("Received request for synthesis")
        request_payload = MessageToDict(request, preserving_proto_field_name=True)
        input_text = request_payload["input_text"]
        voice_id = request_payload["voice"]["voice_id"]
        audio_encoding = request_payload["audio_config"]["audio_encoding"]
        sample_rate_hertz = request_payload["audio_config"]["sample_rate_hertz"]

        # Find the voice configuration
        voice_config = voices.get(voice_id)
        if not voice_config:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid voice id: {voice_id}")
            return

        # Get the corresponding model and speaker ID
        model_name = voice_config["model_name"]
        speaker_id = voice_config["speaker_id"]
        length_scale = voice_config.get("length_scale", 1.0)
        
        model = models.get(model_name)
        if not model:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Model not loaded: {model_name}")
            return

        # Perform inference
        try:
            audio_bytes = model.run_inference(
                text=input_text,
                speaker_id=speaker_id,
                audio_encoding=audio_encoding,
                sample_rate=sample_rate_hertz,
                length_scale=length_scale
            )
            yield piper_pb2.SynthesizeSpeechResponse(audio_content=audio_bytes)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error during synthesis: {str(e)}")
            return

def serve():
    app_port = os.getenv('PORT', 50052)
    
    # Print the port to ensure the server is running on the correct port
    print(f"Starting server on port {app_port}")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    piper_pb2_grpc.add_gwServicer_to_server(gwSpeechServicer(), server)
    server.add_insecure_port(f'[::]:{app_port}')
    server.start()
    
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
