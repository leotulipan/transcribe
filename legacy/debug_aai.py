
import sys
import inspect
import assemblyai
from audio_transcribe.utils.api.assemblyai import AssemblyAIAPI

print(f"AssemblyAI version: {assemblyai.__version__}")
print(f"AssemblyAI file: {assemblyai.__file__}")

print("\nSource of AssemblyAIAPI.__init__:")
print(inspect.getsource(AssemblyAIAPI.__init__))

print("\nAttempting to instantiate AssemblyAIAPI...")
try:
    api = AssemblyAIAPI(api_key="test_key")
    print("Instantiation successful")
except Exception as e:
    print(f"Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
