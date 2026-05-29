; installer/transcribe.iss
;
; Inno Setup 6 script for Audio Transcribe.
;
; Invoked by scripts/build-installer.ps1 with preprocessor defines:
;   /DAppVersion=<x.y.z>            -> stamped into AppVersion and OutputBaseFilename
;   /DSourceDir=<abs path to dist/> -> location of the built transcribe.exe + transcribe-gui.exe
;
; Output: dist/transcribe-setup-v<x.y.z>.exe
;
; Install scope: per-user, no UAC, %LOCALAPPDATA%\Programs\Transcribe.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#ifndef SourceDir
  #define SourceDir "..\dist"
#endif

[Setup]
; AppId is the immutable identifier Windows uses to detect upgrades. NEVER change it.
AppId={{4DF9FFEE-E7A0-4874-9FF5-967FAC17FB80}
AppName=Audio Transcribe
AppVersion={#AppVersion}
AppVerName=Audio Transcribe {#AppVersion}
AppPublisher=Leonard Tulipan
AppPublisherURL=https://github.com/leotulipan/transcribe
AppSupportURL=https://github.com/leotulipan/transcribe/issues
AppUpdatesURL=https://github.com/leotulipan/transcribe/releases
DefaultDirName={localappdata}\Programs\Transcribe
DefaultGroupName=Audio Transcribe
DisableProgramGroupPage=yes
DisableDirPage=no
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#SourceDir}
OutputBaseFilename=transcribe-setup-v{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\transcribe-gui.exe
UninstallDisplayName=Audio Transcribe
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath";     Description: "Add CLI to PATH (transcribe command in any terminal)"; GroupDescription: "CLI integration:"
Name: "startmenuicon"; Description: "Create Start Menu shortcut";                          GroupDescription: "Shortcuts:"
Name: "desktopicon";   Description: "Create Desktop shortcut";                             GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "shellcontext";  Description: "Add ""Transcribe with..."" to right-click menu for audio/video files"; GroupDescription: "Integration:"

[Files]
Source: "{#SourceDir}\transcribe.exe";     DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\transcribe-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";                    DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\LICENSE";                      DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Audio Transcribe";       Filename: "{app}\transcribe-gui.exe"; IconFilename: "{app}\transcribe-gui.exe"; Tasks: startmenuicon
Name: "{group}\Uninstall Audio Transcribe"; Filename: "{uninstallexe}";       Tasks: startmenuicon
Name: "{userdesktop}\Audio Transcribe"; Filename: "{app}\transcribe-gui.exe"; IconFilename: "{app}\transcribe-gui.exe"; Tasks: desktopicon

[Registry]
; Right-click "Transcribe with..." on common media files. Per-user (HKCU) so
; no admin privileges are required. Inno Setup wraps each "Transcribe" key
; with uninsdeletekey so the menu entry vanishes on uninstall.

; --- Audio ---
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp3\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp3\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp3\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.wav\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.wav\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.wav\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.m4a\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.m4a\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.m4a\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.flac\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.flac\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.flac\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ogg\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ogg\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ogg\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.aac\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.aac\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.aac\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

; --- Video ---
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp4\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp4\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mp4\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mov\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mov\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mov\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mkv\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mkv\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.mkv\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webm\shell\Transcribe";         ValueType: string; ValueData: "Transcribe with..."; Tasks: shellcontext; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webm\shell\Transcribe";         ValueType: string; ValueName: "Icon"; ValueData: """{app}\transcribe-gui.exe"""; Tasks: shellcontext
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webm\shell\Transcribe\command"; ValueType: string; ValueData: """{app}\transcribe-gui.exe"" ""%1"""; Tasks: shellcontext

[Code]
const
  ENV_KEY = 'Environment';

// Compare two paths case-insensitively, treating trailing backslashes as equivalent.
function PathsEqual(A, B: string): Boolean;
begin
  Result := CompareText(RemoveBackslashUnlessRoot(A), RemoveBackslashUnlessRoot(B)) = 0;
end;

// Returns True when {app} is not already present in the user's PATH.
function NeedsAddPath(): Boolean;
var
  CurrentPath, AppDir, Remaining, Token: string;
  SepPos: Integer;
begin
  Result := True;
  AppDir := ExpandConstant('{app}');
  if not RegQueryStringValue(HKCU, ENV_KEY, 'Path', CurrentPath) then exit;
  Remaining := CurrentPath;
  while Length(Remaining) > 0 do
  begin
    SepPos := Pos(';', Remaining);
    if SepPos = 0 then
    begin
      Token := Remaining;
      Remaining := '';
    end
    else
    begin
      Token := Copy(Remaining, 1, SepPos - 1);
      Remaining := Copy(Remaining, SepPos + 1, Length(Remaining) - SepPos);
    end;
    if (Token <> '') and PathsEqual(Token, AppDir) then
    begin
      Result := False;
      exit;
    end;
  end;
end;

procedure AddToUserPath();
var
  CurrentPath, AppDir: string;
begin
  AppDir := ExpandConstant('{app}');
  if RegQueryStringValue(HKCU, ENV_KEY, 'Path', CurrentPath) then
  begin
    if not NeedsAddPath() then exit;
    if (Length(CurrentPath) > 0) and (CurrentPath[Length(CurrentPath)] <> ';') then
      CurrentPath := CurrentPath + ';';
    CurrentPath := CurrentPath + AppDir;
  end
  else
    CurrentPath := AppDir;
  RegWriteExpandStringValue(HKCU, ENV_KEY, 'Path', CurrentPath);
end;

procedure RemoveFromUserPath();
var
  CurrentPath, NewPath, AppDir, Token, Remaining: string;
  SepPos: Integer;
begin
  AppDir := ExpandConstant('{app}');
  if not RegQueryStringValue(HKCU, ENV_KEY, 'Path', CurrentPath) then exit;
  NewPath := '';
  Remaining := CurrentPath;
  while Length(Remaining) > 0 do
  begin
    SepPos := Pos(';', Remaining);
    if SepPos = 0 then
    begin
      Token := Remaining;
      Remaining := '';
    end
    else
    begin
      Token := Copy(Remaining, 1, SepPos - 1);
      Remaining := Copy(Remaining, SepPos + 1, Length(Remaining) - SepPos);
    end;
    if (Token <> '') and not PathsEqual(Token, AppDir) then
    begin
      if NewPath <> '' then NewPath := NewPath + ';';
      NewPath := NewPath + Token;
    end;
  end;
  if NewPath <> CurrentPath then
    RegWriteExpandStringValue(HKCU, ENV_KEY, 'Path', NewPath);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if IsTaskSelected('addtopath') then
      AddToUserPath();
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromUserPath();
end;
