from setuptools import setup

# install with 
#   pip install -e .
# not yet working. look more at https://github.com/simonw/llm/blob/main/llm/cli.py

setup(
    name='audio_transcribe_assemblyai',
    version='0.1',
    py_modules=['audio_transcribe_assemblyai'],
    install_requires=[
        'python-dotenv',
        'requests',
        'datetime',
        'argparse',
    ],
    entry_points='''
        [console_scripts]
        audio_transcribe_assemblyai=audio_transcribe_assemblyai:main
    ''',
)