const en = {
    // General UI
    "app_title": "Voice Recorder & Transcriber",
    "app_description": "Record audio, save locally, and transcribe using Cloud APIs.",
    "footer_text": "App made with ♥️ and ✨ in Vienna. Uses browser (local) storage. API Key is not saved in the Cloud.",

    // Buttons & Controls
    "record_button": "Record",
    "stop_button": "Stop",
    "settings_button": "Settings",
    "close_button": "Close",
    "play_button": "Play",
    "pause_button": "Pause",
    "download_button": "Download",
    "transcribe_button": "Transcribe",
    "delete_button": "Delete",
    "copy_button": "Copy to Clipboard",
    "help_button": "Help",

    // Settings
    "settings_title": "Settings",
    "transcription_api": "Transcription API:",
    "recording_settings": "Recording Settings:",
    "recording_limit": "Recording Length Limit (seconds):",
    "recording_limit_note": "Leave empty for automatic limits (FFmpeg: 600s, Gemini: 570s, OpenAI: 140s)",
    "theme": "Theme:",
    "dark_mode": "Dark Mode:",
    "system": "System",
    "dark": "Dark",
    "language": "Interface Language:",

    // API Settings
    "api_key": "API Key:",
    "api_key_note": "Your API key is stored only in your browser's local storage.",
    "language_selection": "Language:",
    "model": "Model:",
    "auto_detect": "Auto-detect",
    
    // FFmpeg
    "ffmpeg_status": "FFmpeg Status:",
    "ffmpeg_enable": "Enable FFmpeg",
    "ffmpeg_loading": "Loading FFmpeg (approx. 30MB)...",
    "ffmpeg_loaded": "FFmpeg loaded successfully",
    "ffmpeg_disabled": "FFmpeg disabled. Audio will not be converted to MP3.",
    "ffmpeg_failed": "Failed to load FFmpeg. Conversion unavailable.",

    // Recording
    "ready_to_record": "Ready to record",
    "recording": "Recording...",
    "processing": "Processing recording...",
    "converting": "Converting audio to MP3...",
    "saving": "Saving recording...",
    "saved": "Recording saved.",
    "drag_drop": "Drag & drop m4a, wav, or mp3 files here to upload",

    // Transcription
    "transcription_title": "Transcription",
    "transcription_placeholder": "Transcription will appear here...",
    "transcribing": "Transcribing...",
    "transcription_complete": "Transcription complete.",
    "copied": "Copied!",
    "copy_failed": "Copy failed.",

    // History
    "history_title": "Recording History",
    "no_history": "No recordings yet.",
    "recorded": "Recorded",
    "size": "Size",
    "duration": "Duration",
    "unknown_duration": "Unknown",
    "tokens": "Tokens",

    // Status messages
    "loading_ffmpeg": "Loading FFmpeg...",
    "mic_access_denied": "Mic access denied.",
    "no_mic_found": "No mic found.",
    "mic_in_use": "Mic in use.",
    "unknown_mic_error": "Unknown mic error.",
    "api_key_needed": "API Key needed for transcription.",
    "api_key_saved": "API Key saved.",
    "api_key_removed": "API Key removed.",
    "no_transcription_api": "No transcription API enabled. Please enable an API in settings.",
    "invalid_audio": "Invalid audio.",

    // Help
    "help_title": "How to Use This App",
    "help_intro": "This voice recorder allows you to record audio from your microphone, save it locally, and transcribe it using various AI services.",
    "help_step1": "1. Set up an API key in Settings (click the gear icon ⚙️ - use the external link 🔗 to get your API key)",
    "help_step2": "2. Click the microphone button to start recording",
    "help_step3": "3. Click again to stop recording",
    "help_step4": "4. Your recording will be saved in the history section",
    "help_step5": "5. Click the transcribe button to convert speech to text",
    "help_apis": "Supported transcription services:",
    "help_api_gemini": "• Google Gemini - Free tier available",
    "help_api_openai": "• OpenAI Whisper - Paid API",
    "help_api_groq": "• Groq - Paid API with free tier",
    "help_api_assembly": "• Assembly AI - Paid API with free tier",
    "help_privacy": "Privacy note: Your recordings stay on your device. API keys are stored in your browser's local storage only.",

    // File Upload
    "no_file_detected": "No file detected.",
    "drop_single_file": "Please drop only one file.",
    "file_type_error": "Please upload only m4a, wav, or mp3 files.",
    "file_too_large": "File too large for",
    "max": "Max",
    "file_uploaded": "File uploaded",
    "file_uploaded_no_api": "File uploaded. Add an API key to enable transcription.",
    "error_processing_file": "Error processing file",
    "starting_transcription": "Starting automatic transcription for recording",
    "api_key_placeholder": "Enter your API key",
    
    // Don't show on startup button
    "dont_show_startup": "Don't show on startup",
    
    // Languages (native names)
    "lang_en": "English",
    "lang_de": "Deutsch (German)",
    
    // Other missing translations
    "loading": "Loading...",
    "error": "Error:",
    "done": "Done",
    "success": "Success",
    "saving": "Saving..."
};

// Export the translations
if (typeof module !== 'undefined' && module.exports) {
    module.exports = en;
} else {
    // For browser use
    window.i18n = window.i18n || {};
    window.i18n.en = en;
} 