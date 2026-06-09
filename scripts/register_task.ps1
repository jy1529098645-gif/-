# Register the daily-track Windows scheduled task. ASCII-only content to avoid encoding loss.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # repo root (parent of \scripts)
$bat  = Join-Path $root "run_daily_track.bat"
if (-not (Test-Path $bat)) { Write-Output "BAT_NOT_FOUND: $bat"; exit 1 }

$taskName = "QuantLabDailyTrack"
$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Daily -At 7:00am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
$t = Get-ScheduledTask -TaskName $taskName
Write-Output ("OK Registered: " + $t.TaskName + " | State=" + $t.State + " | Trigger=Daily 07:00")
Write-Output ("Action -> cmd.exe /c " + $bat)
