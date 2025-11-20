IMPORTANT:
 - After each task make an atomic commit.
 - Add your thinking and details of your implementation  to features.md (max 2-3 paragraphs per feature) so another developer can reproduce your w
 - check off done tasks here

# TASKS
[ ] file/folder as positional (last) argument without "--file" in front (or whatever the current cli best practices are)
[ ] resizing/conversion currently does not happen (at least in assemblyai). Flow should be: extract audio only in current format if at all possible, audio still to big? reencode to flac; still to big: recode to mp3 mono 128kb;  still to big: fail
  [ ] keep converted files until we have a transcription result. only then delete
  [ ] if not yet implemented add a --keep option (this overrides the delete behavior)
[ ] setup screen color instructions do not work. other screens work. example the user sees:
```
transcribe setup
╭────────────────────────╮
│ Audio Transcribe Setup │
╰────────────────────────╯
? Main Menu: (Use arrow keys)
 » Configure assemblyai ([green]Configured[/green])
   Configure elevenlabs ([red]Not Configured[/red])
   Configure groq ([red]Not Configured[/red])
   Configure openai ([red]Not Configured[/red])
   Exit
```