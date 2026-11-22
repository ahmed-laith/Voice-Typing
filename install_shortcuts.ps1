# Creates Desktop and Start Menu shortcuts for Voice Typing
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$wsh = New-Object -ComObject WScript.Shell

# 1. Desktop Shortcut
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopLnk = Join-Path $desktopPath "Voice Typing.lnk"
$s1 = $wsh.CreateShortcut($desktopLnk)
$s1.TargetPath = Join-Path $dir ".venv\Scripts\pythonw.exe"
$s1.Arguments = "voice_typing.py --show"
$s1.WorkingDirectory = $dir
$ico = Join-Path $dir "assets\voice_typing.ico"
if (Test-Path $ico) {
    $s1.IconLocation = $ico
}
$s1.Description = "Voice Typing (F1) - الإملاء الصوتي الذكي"
$s1.Save()
Write-Host "Desktop shortcut created: $desktopLnk"

# 2. Start Menu Programs Shortcut
$programsPath = [Environment]::GetFolderPath("Programs")
$startMenuLnk = Join-Path $programsPath "Voice Typing.lnk"
$s2 = $wsh.CreateShortcut($startMenuLnk)
$s2.TargetPath = Join-Path $dir ".venv\Scripts\pythonw.exe"
$s2.Arguments = "voice_typing.py --show"
$s2.WorkingDirectory = $dir
if (Test-Path $ico) {
    $s2.IconLocation = $ico
}
$s2.Description = "Voice Typing (F1) - الإملاء الصوتي الذكي"
$s2.Save()
Write-Host "Start Menu shortcut created: $startMenuLnk"
