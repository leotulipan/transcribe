const de = {
    // General UI
    "app_title": "Sprachrekorder & Transkription",
    "app_description": "Audio aufnehmen, lokal speichern und mit Cloud-APIs transkribieren.",
    "footer_text": "App mit ♥️ und ✨ in Wien erstellt. Verwendet lokalen Browserspeicher. API-Schlüssel wird nicht in der Cloud gespeichert.",

    // Buttons & Controls
    "record_button": "Aufnehmen",
    "stop_button": "Stopp",
    "settings_button": "Einstellungen",
    "close_button": "Schließen",
    "play_button": "Abspielen",
    "pause_button": "Pause",
    "download_button": "Herunterladen",
    "transcribe_button": "Transkribieren",
    "delete_button": "Löschen",
    "copy_button": "In die Zwischenablage kopieren",
    "help_button": "Hilfe",

    // Settings
    "settings_title": "Einstellungen",
    "transcription_api": "Transkriptions-API:",
    "recording_settings": "Aufnahmeeinstellungen:",
    "recording_limit": "Aufnahmelängenbegrenzung (Sekunden):",
    "recording_limit_note": "Leer lassen für automatische Grenzen (FFmpeg: 600s, Gemini: 570s, OpenAI: 140s)",
    "theme": "Design:",
    "dark_mode": "Dunkelmodus:",
    "system": "System",
    "dark": "Dunkel",
    "language": "Benutzeroberflächen-Sprache:",

    // API Settings
    "api_key": "API-Schlüssel:",
    "api_key_note": "Ihr API-Schlüssel wird nur im lokalen Speicher Ihres Browsers gespeichert.",
    "language_selection": "Sprache:",
    "model": "Modell:",
    "auto_detect": "Automatische Erkennung",
    
    // FFmpeg
    "ffmpeg_status": "FFmpeg-Status:",
    "ffmpeg_enable": "FFmpeg aktivieren",
    "ffmpeg_loading": "FFmpeg wird geladen (ca. 30MB)...",
    "ffmpeg_loaded": "FFmpeg erfolgreich geladen",
    "ffmpeg_disabled": "FFmpeg deaktiviert. Audio wird nicht in MP3 konvertiert.",
    "ffmpeg_failed": "FFmpeg konnte nicht geladen werden. Konvertierung nicht verfügbar.",

    // Recording
    "ready_to_record": "Bereit zur Aufnahme",
    "recording": "Aufnahme...",
    "processing": "Aufnahme wird verarbeitet...",
    "converting": "Audio wird in MP3 konvertiert...",
    "saving": "Aufnahme wird gespeichert...",
    "saved": "Aufnahme gespeichert.",
    "drag_drop": "Ziehen Sie m4a-, wav- oder mp3-Dateien hierher, um sie hochzuladen",

    // Transcription
    "transcription_title": "Transkription",
    "transcription_placeholder": "Transkription erscheint hier...",
    "transcribing": "Transkribiere...",
    "transcription_complete": "Transkription abgeschlossen.",
    "copied": "Kopiert!",
    "copy_failed": "Kopieren fehlgeschlagen.",

    // History
    "history_title": "Aufnahmehistorie",
    "no_history": "Noch keine Aufnahmen.",
    "recorded": "Aufgenommen",
    "size": "Größe",
    "duration": "Dauer",
    "unknown_duration": "Unbekannt",
    "tokens": "Tokens",

    // Status messages
    "loading_ffmpeg": "FFmpeg wird geladen...",
    "mic_access_denied": "Mikrofonzugriff verweigert.",
    "no_mic_found": "Kein Mikrofon gefunden.",
    "mic_in_use": "Mikrofon wird bereits verwendet.",
    "unknown_mic_error": "Unbekannter Mikrofonfehler.",
    "api_key_needed": "API-Schlüssel für die Transkription erforderlich.",
    "api_key_saved": "API-Schlüssel gespeichert.",
    "api_key_removed": "API-Schlüssel entfernt.",
    "no_transcription_api": "Keine Transkriptions-API aktiviert. Bitte aktivieren Sie eine API in den Einstellungen.",
    "invalid_audio": "Ungültige Audiodatei.",

    // Help
    "help_title": "Anleitung zur Nutzung dieser App",
    "help_intro": "Mit diesem Sprachrekorder können Sie Audio über Ihr Mikrofon aufnehmen, lokal speichern und mit verschiedenen KI-Diensten transkribieren.",
    "help_step1": "1. Richten Sie einen API-Schlüssel in den Einstellungen ein (klicken Sie auf das Zahnrad-Symbol ⚙️ - verwenden Sie den externen Link 🔗 um Ihren API-Schlüssel zu erhalten)",
    "help_step2": "2. Klicken Sie auf die Mikrofontaste, um die Aufnahme zu starten",
    "help_step3": "3. Klicken Sie erneut, um die Aufnahme zu beenden",
    "help_step4": "4. Ihre Aufnahme wird im Verlaufsbereich gespeichert",
    "help_step5": "5. Klicken Sie auf die Transkriptionstaste, um Sprache in Text umzuwandeln",
    "help_apis": "Unterstützte Transkriptionsdienste:",
    "help_api_gemini": "• Google Gemini - Kostenloses Kontingent verfügbar",
    "help_api_openai": "• OpenAI Whisper - Kostenpflichtige API",
    "help_api_groq": "• Groq - Kostenpflichtige API mit kostenlosem Kontingent",
    "help_api_assembly": "• Assembly AI - Kostenpflichtige API mit kostenlosem Kontingent",
    "help_privacy": "Datenschutzhinweis: Ihre Aufnahmen bleiben auf Ihrem Gerät. API-Schlüssel werden nur im lokalen Speicher Ihres Browsers gespeichert.",

    // File Upload
    "no_file_detected": "Keine Datei erkannt.",
    "drop_single_file": "Bitte nur eine Datei ablegen.",
    "file_type_error": "Bitte nur m4a-, wav- oder mp3-Dateien hochladen.",
    "file_too_large": "Datei zu groß für",
    "max": "Maximum",
    "file_uploaded": "Datei hochgeladen",
    "file_uploaded_no_api": "Datei hochgeladen. Fügen Sie einen API-Schlüssel hinzu, um die Transkription zu aktivieren.",
    "error_processing_file": "Fehler bei der Verarbeitung der Datei",
    "starting_transcription": "Automatische Transkription für Aufnahme wird gestartet",
    "api_key_placeholder": "Geben Sie Ihren API-Schlüssel ein",
    
    // Don't show on startup button
    "dont_show_startup": "Beim Start nicht anzeigen",
    
    // Languages (native names)
    "lang_en": "English (Englisch)",
    "lang_de": "Deutsch",
    
    // Other missing translations
    "loading": "Wird geladen...",
    "error": "Fehler:",
    "done": "Fertig",
    "success": "Erfolg"
};

// Export the translations
if (typeof module !== 'undefined' && module.exports) {
    module.exports = de;
} else {
    // For browser use
    window.i18n = window.i18n || {};
    window.i18n.de = de;
} 