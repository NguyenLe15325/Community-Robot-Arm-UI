; Inno Setup Script for Community Robot Arm UI
; Produces a professional Windows installer (.exe)
; Download Inno Setup from: https://jrsoftware.org/isinfo.php

[Setup]
AppName=Community Robot Arm UI
AppVersion=1.0.0
AppPublisher=NguyenLe15325
AppPublisherURL=https://github.com/NguyenLe15325/Community-Robot-Arm-UI
DefaultDirName={autopf}\CommunityRobotArmUI
DefaultGroupName=Community Robot Arm UI
UninstallDisplayIcon={app}\CommunityRobotArmUI.exe
OutputDir=installer_output
OutputBaseFilename=CommunityRobotArmUI-Setup
SetupIconFile=assets\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle everything from PyInstaller dist folder
Source: "dist\CommunityRobotArmUI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Community Robot Arm UI"; Filename: "{app}\CommunityRobotArmUI.exe"
Name: "{group}\Uninstall Community Robot Arm UI"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Community Robot Arm UI"; Filename: "{app}\CommunityRobotArmUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CommunityRobotArmUI.exe"; Description: "{cm:LaunchProgram,Community Robot Arm UI}"; Flags: nowait postinstall skipifsilent
