$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$v5 = Join-Path $root 'data\v5'
$queueLog = Join-Path $v5 'continuation_queue.log'

function Log-Line([string]$message) {
    Add-Content -LiteralPath $queueLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
}

function Run-Step([string]$name, [string]$script, [string[]]$arguments = @()) {
    $log = Join-Path $v5 ($name + '.queue.log')
    Log-Line "START ${name}: py -3.12 $script $($arguments -join ' ')"
    & py -3.12 $script @arguments *> $log
    $code = $LASTEXITCODE
    Log-Line "END $name exit_code=$code log=$log"
    return $code
}

Log-Line 'Resuming at model experiments after corrected E1 summary; P1 gates/diagnostics are complete.'
Log-Line 'E2 smoke already passed: B/M0/logreg/E0, 240490 score rows, 3342 positives, 6 crash events.'
$e2 = Run-Step 'e2_full' 'v5_9_e2.py' @('--timebox-min', '90')
$e2Summary = Run-Step 'e2_summary' 'v5_9_summarize_e2.py'
if ($e2Summary -ne 0) { Log-Line 'STOP: E2 summary failed; E3/E4 were not launched.'; exit 2 }

$e3Smoke = Run-Step 'e3_smoke' 'v5_8_e3_deep.py' @('--smoke')
if ($e3Smoke -ne 0) { Log-Line 'STOP: E3 smoke failed; E4 was not launched.'; exit 3 }
$e3 = Run-Step 'e3_full' 'v5_8_e3_deep.py' @('--timebox-min', '150')
$e4Smoke = Run-Step 'e4_smoke' 'v5_10_e4_labels.py' @('--smoke')
if ($e4Smoke -ne 0) { Log-Line 'STOP: E4 smoke failed.'; exit 4 }
$e4 = Run-Step 'e4_minimal' 'v5_10_e4_labels.py' @('--timebox-min', '90')
$episodes = Run-Step 'episode_table' 'v5_11_episode_table.py'
Log-Line 'Configured P1 model queue reached its end; E5-E7 P2 remain unrun.'
