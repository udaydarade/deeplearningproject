$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$v5 = Join-Path $root 'data\v5'
$queueLog = Join-Path $v5 'continuation_queue.log'
New-Item -ItemType Directory -Force -Path $v5 | Out-Null

function Log-Line([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -LiteralPath $queueLog -Value $line
}

function Run-Step([string]$name, [string]$script, [string[]]$arguments = @()) {
    $log = Join-Path $v5 ($name + '.queue.log')
    Log-Line "START ${name}: py -3.12 $script $($arguments -join ' ')"
    & py -3.12 $script @arguments *> $log
    $code = $LASTEXITCODE
    Log-Line "END $name exit_code=$code log=$log"
    return $code
}

Log-Line 'Queue started; waiting for corrected TDA, then overlapping quality/dynamics work with E1 (V6 untouched).'
foreach ($processId in @(9744)) {
    while (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
    Log-Line "Existing process $processId exited."
}

$oldQuality = @('quality_gate_results.csv','quality_leak_test.csv','quality_scale_test.csv',
                'tda_quality_by_column.csv','tda_quality_by_group.csv')
$archive = Join-Path $v5 'archive_quality_pre_corrected_rebuild'
New-Item -ItemType Directory -Force -Path $archive | Out-Null
foreach ($name in $oldQuality) {
    $source = Join-Path $v5 $name
    if (Test-Path -LiteralPath $source) {
        $target = Join-Path $archive $name
        if (Test-Path -LiteralPath $target) {
            $target = Join-Path $archive (([IO.Path]::GetFileNameWithoutExtension($name)) + '_prior.csv')
        }
        Move-Item -LiteralPath $source -Destination $target
    }
}
Log-Line "Archived pre-rebuild quality outputs under $archive before recomputing gates."
$universe = Run-Step 'tda_universe_enforcement' 'v5_2_enforce_universe.py'
$gates = Run-Step 'tda_quality_gates' 'v5_6_quality_gates.py'
if ($universe -ne 0 -or $gates -ne 0) {
    Log-Line 'STOP: topology postprocessing failed; downstream V5 comparisons were not launched.'
    exit 1
}
$gateRows = Import-Csv (Join-Path $v5 'quality_gate_results.csv')
$failedGate = @($gateRows | Where-Object { $_.pass -ne 'True' })
if ($failedGate.Count -gt 0) {
    Log-Line "STOP: quality gates failed ($($failedGate.gate -join ', ')); diagnose before downstream comparisons."
    while (Get-Process -Id 13816 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 15 }
    Run-Step 'e1_offbudget_diagnostic' 'v5_1_diagnose_offbudget.py' | Out-Null
    Run-Step 'e1_summary' 'v5_7_summarize_e1.py' | Out-Null
    exit 2
}

$pilot = Run-Step 'dynamics_pilot' 'v5_5_build_dynamics.py' @('--pilot')
$dynamics = Run-Step 'dynamics_build' 'v5_5_build_dynamics.py' @('--landscape-timebox-min', '45')
$diagnostics = Run-Step 'feature_diagnostics' 'v5_6_diagnostics.py'

while (Get-Process -Id 13816 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 15 }
Log-Line 'E1 process exited; producing its diagnosis and validation-only summary.'
Run-Step 'e1_offbudget_diagnostic' 'v5_1_diagnose_offbudget.py' | Out-Null
Run-Step 'e1_summary' 'v5_7_summarize_e1.py' | Out-Null

$e2Smoke = Run-Step 'e2_smoke' 'v5_9_e2.py' @('--smoke')
if ($e2Smoke -eq 0) {
    $e2 = Run-Step 'e2_full' 'v5_9_e2.py' @('--timebox-min', '90')
    $e2Summary = Run-Step 'e2_summary' 'v5_9_summarize_e2.py'
}

$e3Smoke = Run-Step 'e3_smoke' 'v5_8_e3_deep.py' @('--smoke')
if ($e3Smoke -eq 0) {
    $e3 = Run-Step 'e3_full' 'v5_8_e3_deep.py' @('--timebox-min', '150')
}

$e4Smoke = Run-Step 'e4_smoke' 'v5_10_e4_labels.py' @('--smoke')
if ($e4Smoke -eq 0) {
    $e4 = Run-Step 'e4_minimal' 'v5_10_e4_labels.py' @('--timebox-min', '90')
}

$episodes = Run-Step 'episode_table' 'v5_11_episode_table.py'
Log-Line 'Queue reached end of configured P1 experiments. E5-E7 P2 remain pending for separate assessment.'
