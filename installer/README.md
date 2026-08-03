# Windows installer

Scientific RAG is a Python/Streamlit application, not a compiled native program,
so "the installer" gives you a real Windows setup experience while still running
on the Python interpreter you (or the installer) put on the machine. Two options
are provided; use whichever fits:

## Option A - `install.bat` (no build step, use this by default)

1. Install **Python 3.10+** from <https://www.python.org/downloads/> if it is not
   already present. During its setup, tick **"Add python.exe to PATH"**.
2. Download or `git clone` this repository anywhere, e.g. `C:\Apps\scientific_rag`.
3. Double-click **`installer\install.bat`** (or run it from a Command Prompt).

It will, entirely inside the project folder and without Administrator rights:

- verify Python 3.10+ is available;
- create an isolated virtual environment in `.venv`;
- install every dependency from `src\requirements.txt`;
- create `.env` from `.env.example` if it does not exist yet;
- initialise the local SQLite database and application folders;
- write **`Run Scientific RAG.bat`** in the project root;
- optionally create a Desktop shortcut.

Launch the app afterwards with `Run Scientific RAG.bat` or the Desktop shortcut.
Run **`installer\uninstall.bat`** to remove the virtual environment and the
generated launcher/shortcut again (your chats, settings and cached papers under
`%LOCALAPPDATA%\ScientificRAG` are left untouched; delete that folder yourself
to erase them too).

## Option B - `scientific_rag.iss` (a traditional `Setup.exe` wizard)

For a double-click **Setup.exe** with a Start Menu entry, licence/progress
pages and a proper uninstall entry in "Add or Remove Programs", compile the
included [Inno Setup](https://jrsoftware.org/isinfo.php) script on a Windows
machine:

```powershell
choco install innosetup -y          # or download the Inno Setup installer manually
ISCC.exe installer\scientific_rag.iss
```

This produces `installer\dist\ScientificRAG-Setup-1.0.0.exe`. Running that
Setup:

1. Copies the application source tree to `%ProgramFiles%\Scientific RAG` (per
   the current user, no elevation required).
2. Runs `install.bat` automatically to create the virtual environment and
   install dependencies (needs the same Python 3.10+ prerequisite and internet
   access on first run).
3. Creates Start Menu and, optionally, Desktop shortcuts.
4. Offers to launch the app immediately after setup finishes.

Uninstalling through "Add or Remove Programs" removes the copied files, the
virtual environment and `.env`; it never touches
`%LOCALAPPDATA%\ScientificRAG` (your data).

> Why not one single bundled `.exe` with Python inside? Freezing the whole
> app (PyInstaller/Nuitka) is possible but must be built *on Windows* and
> bundles ~300&nbsp;MB of interpreter and ML/PDF libraries per release. The
> two options above install in seconds, always match the pinned
> `requirements.txt`, and are trivial to keep up to date - which is why they
> are what this repository ships.
