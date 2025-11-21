# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Support for multiple transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI)
- Multiple output formats (text, SRT, word-level SRT, DaVinci Resolve optimized)
- Interactive setup wizard for API key configuration
- Batch file templates for easy launching
- Standalone Windows executable

### Changed
- Moved from individual API scripts to unified CLI tool

### Security
- API keys now stored securely in user profile directory
- Removed legacy .env files from repository

[Unreleased]: https://github.com/yourusername/audio-transcribe/compare/v0.1.0...HEAD

