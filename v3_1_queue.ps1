$ErrorActionPreference = 'Continue'
function Stop-Task($p) {
  $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $p.Id }
  foreach ($child in $children) { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
$project = (Resolve-Path '.').Path
Set-Location $project
$log = Join-Path $project 'data\v3\v31_queue.log'
Start-Transcript -Path $log -Append | Out-Null
Write-Output 'V3.1 queue waiting for the currently running Step 2 baseline process.'
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'py.exe' -and $_.CommandLine -like '*v3_1_train.py --step 2 --sets B1,B2,B3*'
} | Select-Object -First 1
if ($active) { Wait-Process -Id $active.ProcessId }
Write-Output 'Step 2 baselines ended; starting F0 logreg + GBM (20 minute timebox).'
$out = Join-Path $project 'data\v3\v31_step2_f0.stdout.log'
$err = Join-Path $project 'data\v3\v31_step2_f0.stderr.log'
$proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_train.py','--step','2','--sets','F0','--models','logreg,gbm') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Wait-Process -Id $proc.Id -Timeout 1200 -ErrorAction SilentlyContinue
$proc.Refresh()
if (!$proc.HasExited) { Write-Output 'F0 timebox reached; terminating runner. Per-fold scores remain saved.'; Stop-Task $proc }
py -3.12 v3_1_summarize.py 2
Write-Output 'Starting Step 3 classical ladder (35 minute timebox).'
$out = Join-Path $project 'data\v3\v31_step3.stdout.log'
$err = Join-Path $project 'data\v3\v31_step3.stderr.log'
$proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_run_steps.py','3') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Wait-Process -Id $proc.Id -Timeout 2100 -ErrorAction SilentlyContinue
$proc.Refresh()
if (!$proc.HasExited) { Write-Output 'Step 3 timebox reached; terminating runner; completed score files remain saved.'; Stop-Task $proc }
if (Test-Path (Join-Path $project 'data\v3\v31_best_classical.json')) {
  Write-Output 'Starting Step 4 TDA blocks (60 minute timebox).'
  $out = Join-Path $project 'data\v3\v31_step4.stdout.log'
  $err = Join-Path $project 'data\v3\v31_step4.stderr.log'
  $proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_run_steps.py','4') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Wait-Process -Id $proc.Id -Timeout 3600 -ErrorAction SilentlyContinue
  $proc.Refresh()
  if (!$proc.HasExited) { Write-Output 'Step 4 timebox reached; terminating runner; completed score files remain saved.'; Stop-Task $proc }
  py -3.12 v3_1_summarize.py 4
  Write-Output 'Starting Step 5 scale/homology ablations (30 minute timebox).'
  $out = Join-Path $project 'data\v3\v31_step5.stdout.log'
  $err = Join-Path $project 'data\v3\v31_step5.stderr.log'
  $proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_more.py','5','--resume') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Wait-Process -Id $proc.Id -Timeout 1800 -ErrorAction SilentlyContinue
  $proc.Refresh(); if (!$proc.HasExited) { Stop-Task $proc; Write-Output 'Step 5 timebox reached.' }
  py -3.12 v3_1_summarize.py 5
  Write-Output 'Starting Step 6 asset-universe/history comparison (30 minute timebox).'
  $out = Join-Path $project 'data\v3\v31_step6.stdout.log'
  $err = Join-Path $project 'data\v3\v31_step6.stderr.log'
  $proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_more.py','6') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Wait-Process -Id $proc.Id -Timeout 1800 -ErrorAction SilentlyContinue
  $proc.Refresh(); if (!$proc.HasExited) { Stop-Task $proc; Write-Output 'Step 6 timebox reached.' }
  py -3.12 v3_1_summarize.py 6
  Write-Output 'Starting Step 7 label sensitivity (25 minute timebox).'
  $out = Join-Path $project 'data\v3\v31_step7.stdout.log'
  $err = Join-Path $project 'data\v3\v31_step7.stderr.log'
  $proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_more.py','7','--resume') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Wait-Process -Id $proc.Id -Timeout 1500 -ErrorAction SilentlyContinue
  $proc.Refresh(); if (!$proc.HasExited) { Stop-Task $proc; Write-Output 'Step 7 timebox reached.' }
  py -3.12 v3_1_summarize.py 7
}
if (Test-Path (Join-Path $project 'data\v3\v31_F4c.json')) {
  Write-Output 'Starting Step 8 blocked CV and final-model export (30 minute timebox).'
  $out = Join-Path $project 'data\v3\v31_step8.stdout.log'
  $err = Join-Path $project 'data\v3\v31_step8.stderr.log'
  $proc = Start-Process -FilePath 'py.exe' -ArgumentList @('-3.12','v3_1_final.py') -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Wait-Process -Id $proc.Id -Timeout 1800 -ErrorAction SilentlyContinue
  $proc.Refresh(); if (!$proc.HasExited) { Stop-Task $proc; Write-Output 'Step 8 timebox reached.' }
}
py -3.12 v3_1_pack_scores.py
Write-Output 'V3.1 P1 queue finished or reached its hard timeboxes.'
Stop-Transcript | Out-Null

