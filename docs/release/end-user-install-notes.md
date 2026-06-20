# End-user install notes (unsigned builds)

Audio Transcribe binaries are **not code-signed** yet, so Windows and macOS
show a one-time security warning. Paste the relevant block below into each
GitHub Release description so downloaders know what to do.

---

## Windows

1. Download `transcribe-setup-vX.Y.Z.exe` and run it.
2. If **SmartScreen** says *"Windows protected your PC"*, click
   **More info → Run anyway**. (The installer is per-user and needs no
   admin rights.)
3. The installer adds a Start Menu shortcut, the optional right-click
   *"Transcribe with…"* entry, and `transcribe` on your PATH.

Prefer no installer? Grab `transcribe-windows-amd64-vX.Y.Z.zip`, unzip it,
and run `transcribe.exe` (CLI/TUI) or `transcribe-gui.exe` (GUI) directly.

---

## macOS

The macOS download is a single `transcribe` binary (CLI + TUI + GUI in one).

1. Download the tarball for your Mac:
   - Apple Silicon (M-series): `transcribe-macos-arm64-vX.Y.Z.tar.gz`
   - Intel: `transcribe-macos-amd64-vX.Y.Z.tar.gz`
2. Extract and move it onto your PATH:
   ```bash
   tar -xzf transcribe-macos-arm64-vX.Y.Z.tar.gz
   sudo mv transcribe /usr/local/bin/
   ```
3. Clear the Gatekeeper quarantine flag (required because the binary is
   unsigned), then run it:
   ```bash
   xattr -dr com.apple.quarantine /usr/local/bin/transcribe
   transcribe --version
   ```
   Alternatively, the first time you run it, right-click the binary in
   Finder → **Open** → **Open** to approve it once.
4. Run it:
   - `transcribe "audio.mp3"` — command line
   - `transcribe` — terminal UI (TUI)
   - `transcribe --ui=gui` — graphical (Fyne) window

Install **ffmpeg** for video/large files: `brew install ffmpeg`.

---

## Verifying a download (optional)

Each artifact ships a `.sha256` sidecar:

```bash
# macOS / Linux
shasum -a 256 -c transcribe-macos-arm64-vX.Y.Z.tar.gz.sha256
```

```powershell
# Windows
(Get-FileHash transcribe-setup-vX.Y.Z.exe -Algorithm SHA256).Hash
# compare against the .sha256 file's contents
```
