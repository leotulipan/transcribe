#!/usr/bin/env bash
# Build & install helper for audio-transcribe
# Removes stray desktop.ini files, then installs via uv

set -e
cd "$(dirname "$0")"

echo "Removing desktop.ini files..."
rm -f desktop.ini

echo "Installing audio-transcribe (editable)..."
uv tool install --editable .

echo ""
transcribe --version
echo "Done."
