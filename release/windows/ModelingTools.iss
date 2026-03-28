[Setup]
AppName=Modeling Tools
AppVersion=0.1.0
DefaultDirName={localappdata}\Modeling-Tools
DefaultGroupName=Modeling Tools
AppPublisherURL=https://github.com/alexbcreiner0/Modeling-Tools
AppSupportURL=https://github.com/alexbcreiner0/Modeling-Tools
AppUpdatesURL=https://github.com/alexbcreiner0/Modeling-Tools
UninstallDisplayIcon={app}\assets\icon.ico
OutputDir=.
OutputBaseFilename=ModelingTools-Setup
CreateUninstallRegKey=yes

[Files]
Source: "install.ps1"; DestDir: "{app}";
Source: "launcher.ps1"; DestDir: "{app}"
Source: "uninstall.ps1"; DestDir: "{app}"
Source: "..\..\src\*"; DestDir: "{app}\src"; Flags: recursesubdirs; Excludes: "__pycache__\*,*.pyc,*.pyo"
Source: "..\..\defaults\*"; DestDir: "{app}\defaults"; Flags: recursesubdirs
Source: "..\..\README.md"; DestDir: "{app}"
Source: "..\..\pyproject.toml"; DestDir: "{app}"
Source: "..\..\assets\icon.ico"; DestDir: "{app}\assets"
Source: "launcher.pyw"; DestDir: "{app}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; Flags: waituntilterminated

[Icons]
Name: "{group}\Modeling Tools"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\icon.ico"
Name: "{commondesktop}\Modeling Tools"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\icon.ico"

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\uninstall.ps1"""; Flags: waituntilterminated