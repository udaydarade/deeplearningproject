$ErrorActionPreference='Continue'
$project=(Resolve-Path '.').Path
Set-Location $project
$queue=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq 32556 }
if ($queue) { Wait-Process -Id 32556 }
py -3.12 v3_1_pack_scores.py
foreach ($step in @('2','3','4','5','6','7')) {
  if (Test-Path (Join-Path $project "data\v3\v31_step$step.csv")) { py -3.12 v3_1_log_step.py $step }
}
'Post-queue score packing and experiment-log updates complete.' | Add-Content -Encoding utf8 (Join-Path $project 'data\v3\v31_queue.log')
