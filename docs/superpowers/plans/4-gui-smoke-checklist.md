# GUI Manual Smoke Checklist

Fyne does not have a stable headless test harness on Windows + CGO. This
checklist replaces automated tests for the `internal/delivery/gui` package.
Run it after every build that touches the GUI layer.

## Setup

1. Build both binaries: `./scripts/build.ps1`
2. Ensure at least one API key is present in
   `%LOCALAPPDATA%\transcribe\config.toml`.
3. Have a short audio or video file ready (< 1 MB recommended for speed).

## Checklist

### Launch

- [ ] `bin/transcribe-gui.exe` launches from Explorer without opening a console
      window.
- [ ] `bin/transcribe.exe` (double-click from Explorer) opens the GUI window
      (no args = GUI mode on Windows).
- [ ] Window title reads "Transcribe" and is approximately 700 × 500 px.

### Provider / Model dropdowns

- [ ] Provider dropdown lists only providers whose API keys are configured.
- [ ] Selecting a provider immediately populates the model dropdown with that
      provider's models.
- [ ] Switching provider clears and repopulates the model dropdown.

### File browser

- [ ] Clicking "Browse…" opens a file-open dialog.
- [ ] Selecting a file sets the path entry text to the selected file's path.
- [ ] Cancelling the dialog leaves the path entry unchanged.

### Validation

- [ ] Clicking "Start" with no file shows "Pick a file" info dialog.
- [ ] Clicking "Start" with no format checked shows "Pick a format" info dialog.
- [ ] Clicking "Start" with no provider (all keys removed) shows "Pick a
      provider" info dialog.

### Transcription

- [ ] With a Groq key configured and a small `.mp3`, clicking "Start" starts
      the indeterminate progress bar and logs `[probing] …` in the log pane.
- [ ] Progress bar transitions from indeterminate to determinate when chunked
      transcription reports a percentage.
- [ ] On completion, a "Done" dialog appears with a preview of the transcript.
- [ ] A `.txt` file (and `.srt` if checked) appears next to the source audio
      file.

### Cancel

- [ ] Clicking "Cancel" during a job stops it; the buttons return to their
      idle state.
- [ ] Clicking "Start" again after a cancel successfully starts a new job.

### Window close mid-job

- [ ] Closing the window while a job is in flight shows a "Cancel running job?"
      confirmation dialog.
- [ ] Confirming closes the window; dismissing leaves the job running.

### Settings window

- [ ] Clicking "Settings…" opens the settings window.
- [ ] All six API key fields are password entries (text is masked).
- [ ] Entering a new key and clicking "Save" writes to
      `%LOCALAPPDATA%\transcribe\config.toml`.
- [ ] After restarting `transcribe-gui.exe`, the new key is picked up and the
      provider appears in the dropdown.
- [ ] Clicking "Close" dismisses the settings window without saving.

## Notes

- These checks are manual because Fyne's rendering pipeline requires a real
  display and CGO, making headless automation fragile on Windows.
- If any check fails, file a bug with the step that failed and the error shown
  in the log pane or stderr (run `bin/transcribe.exe --ui=gui` from a terminal
  to see stderr output).
