; ===========================================================================
;  Scientific RAG - Inno Setup script
;
;  Produces a traditional Windows "Setup.exe" wizard that installs the
;  application source tree under Program Files, creates Start Menu / Desktop
;  shortcuts, and runs the bundled installer (install.bat) once to create the
;  Python virtual environment and install dependencies.
;
;  Requirements to BUILD this installer (not to run the app):
;    - Windows with Inno Setup 6 installed: https://jrsoftware.org/isinfo.php
;    - Compile with:  ISCC.exe installer\scientific_rag.iss
;    - The resulting Setup EXE is written to installer\dist\.
;
;  Requirements the END USER still needs on their machine:
;    - Python 3.10+ (the installer checks for it and stops with instructions
;      if it is missing - the wizard does not bundle a Python runtime).
;    - Internet access the first time, to install Python dependencies.
; ===========================================================================

#define AppName "Scientific RAG"
#define AppVersion "1.0.0"
#define AppPublisher "Scientific RAG"
#define AppExeName "Run Scientific RAG.bat"

[Setup]
AppId={{6F1B7C2E-6D9B-4C7A-9F0D-3B7B2E7B9A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=ScientificRAG-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern

; The application manages its own per-user data under
; %LOCALAPPDATA%\ScientificRAG (see src/core/constants.py); nothing here
; requires elevated privileges.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Ship the whole project tree except developer/runtime artefacts that are
; recreated locally by install.bat (virtual env, caches, VCS metadata).
Source: "..\app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\init_database.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc,.pytest_cache"
Source: "..\document\*"; DestDir: "{app}\document"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "install.bat"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "uninstall.bat"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\Run Scientific RAG.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Run Scientific RAG.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Sets up the virtual environment, installs dependencies, creates .env and
; the database, and writes "Run Scientific RAG.bat" inside {app}.
Filename: "{cmd}"; Parameters: "/c ""{app}\installer\install.bat"" /inno"; \
    WorkingDir: "{app}"; StatusMsg: "Setting up the Python environment (this can take a few minutes)..."; \
    Flags: waituntilterminated

Filename: "{app}\Run Scientific RAG.bat"; Description: "Launch {#AppName} now"; \
    WorkingDir: "{app}"; Flags: postinstall skipifsilent nowait

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: files; Name: "{app}\.env"
