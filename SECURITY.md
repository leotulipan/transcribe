# Security Policy

## Supported Versions

We provide security updates for the latest release version of Audio Transcribe.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please report it privately by:

1. Email the maintainers directly (if you have contact information)
2. Open a private security advisory on GitHub (if you have access)
3. Contact the repository owner through GitHub

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

We will acknowledge receipt of your report within 48 hours and provide an update on the status of the vulnerability within 7 days.

## Security Best Practices

When using Audio Transcribe:

- **Never commit API keys to version control** - Use the built-in setup wizard to store keys securely
- **Keep your API keys private** - Don't share them publicly or in screenshots
- **Use environment variables** - The tool stores keys in your user profile directory, not in the project directory
- **Review dependencies** - We use well-maintained packages, but always review what you're installing
- **Update regularly** - Keep the tool updated to receive security patches

## API Key Security

Audio Transcribe stores API keys in:
- **Windows**: `%LOCALAPPDATA%\audio_transcribe\.env`
- **Linux/Mac**: `~/.audio_transcribe/.env`

These files are:
- Stored in your user profile (not in the project directory)
- Not tracked by git (excluded via .gitignore)
- Only readable by your user account (on Unix systems)

If you suspect your API keys have been compromised:
1. Revoke the keys immediately from the respective API provider
2. Generate new keys
3. Run `transcribe.exe setup` to update your configuration

