# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-XX

### Added
- Public release preparation
- Batch templates directory with ready-to-use .bat files, icons, and shortcuts
- GitHub Actions workflows for automated CI/CD and releases
- Comprehensive documentation (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY)
- GitHub issue and pull request templates
- Release checklist documentation
- Enhanced build script that creates zip archives with executables

### Changed
- Complete README rewrite focused on end-user experience
- Moved legacy documentation to separate `legacy-docs` branch
- Improved build process to include LICENSE and README in release artifacts

### Security
- API keys stored securely in user profile directory (not in project directory)
- Removed legacy .env files from repository

## [0.1.4] - Previous versions

### Added
- Initial public release
- Support for multiple transcription APIs (AssemblyAI, ElevenLabs, Groq, OpenAI)
- Multiple output formats (text, SRT, word-level SRT, DaVinci Resolve optimized)
- Interactive setup wizard for API key configuration
- Standalone Windows executable

### Changed
- Moved from individual API scripts to unified CLI tool

[Unreleased]: https://github.com/leotulipan/transcribe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/leotulipan/transcribe/compare/v0.1.4...v0.2.0

