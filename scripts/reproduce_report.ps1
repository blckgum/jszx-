param(
    [string]$Python = "",
    [string]$Device = "0",
    [string]$OutputRoot = "runs/analysis",
    [switch]$OfflineOnly,
    [switch]$SkipDemos,
    [switch]$SkipBenchmarks,
    [switch]$SkipCharts,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-PythonExecutable {
    param([string]$Requested)
    if ($Requested -and $Requested.Trim()) {
        return $Requested.Trim()
    }
    $DefaultEnvPy = "D:\AI\plant_yolo_env\Scripts\python.exe"
    if (Test-Path $DefaultEnvPy) {
        return $DefaultEnvPy
    }
    return "python"
}

function Format-CommandArg {
    param([string]$Arg)
    if ($Arg -match '[\s"`$]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

function Invoke-PyStep {
    param(
        [string]$Label,
        [string]$PyExe,
        [string[]]$PyArgs
    )
    $rendered = ($PyArgs | ForEach-Object { Format-CommandArg $_ }) -join " "
    Write-Host "[$Label] $PyExe $rendered"
    if ($DryRun) {
        return
    }
    & $PyExe @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Label' failed with exit code $LASTEXITCODE"
    }
}

$PyExe = Resolve-PythonExecutable -Requested $Python

Push-Location $RepoRoot
try {
    $expDir = Join-Path $OutputRoot "experiments"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $expDir | Out-Null
    }

    $demos = @(
        @{ Name = "sherbrooke_intersection"; Margin = "2.5" },
        @{ Name = "rouen_crosswalk";         Margin = "1.0" },
        @{ Name = "stmarc_mixed_vru";        Margin = "1.0" }
    )

    if (-not $OfflineOnly -and -not $SkipDemos) {
        $weightsPath = "checkpoints/yolo/yolov8n.pt"
        if (-not (Test-Path $weightsPath)) {
            throw "Missing weights file: $weightsPath"
        }
        foreach ($d in $demos) {
            $videoPath = "data/demo_videos/$($d.Name).mp4"
            $calibPath = "data/demo_videos/calib_$($d.Name).yaml"
            if (-not (Test-Path $videoPath)) {
                throw "Missing demo video: $videoPath"
            }
            if (-not (Test-Path $calibPath)) {
                throw "Missing calibration file: $calibPath"
            }
        }

        foreach ($d in $demos) {
            $outDir = Join-Path $OutputRoot "demo_$($d.Name)"
            if (-not $DryRun -and (Test-Path $outDir)) {
                Remove-Item -Recurse -Force $outDir
            }
            Invoke-PyStep -Label "demo-$($d.Name)" -PyExe $PyExe -PyArgs @(
                "-m", "analysis",
                "--source", "data/demo_videos/$($d.Name).mp4",
                "--calib", "data/demo_videos/calib_$($d.Name).yaml",
                "--output", $outDir,
                "--device", $Device,
                "--iou", "0.65",
                "--imgsz", "960",
                "--zone-margin", $d.Margin,
                "--max-frames", "360",
                "--speed-model", "ctrv",
                "--conflict-model", "obb_ssm",
                "--enable-stitching",
                "--include-cyclist-vru",
                "--compensate-anchor-bias"
            )
            Invoke-PyStep -Label "schema-$($d.Name)" -PyExe $PyExe -PyArgs @(
                "-m", "analysis.schema", "--run", $outDir
            )
        }
    }

    $ablationOut = Join-Path $expDir "ablation.json"
    Invoke-PyStep -Label "ablation" -PyExe $PyExe -PyArgs @(
        "-m", "analysis.ablation", "--output", $ablationOut
    )

    $scenariosOut = Join-Path $expDir "scenarios.json"
    Invoke-PyStep -Label "scenarios" -PyExe $PyExe -PyArgs @(
        "-m", "analysis.scenarios", "--output", $scenariosOut
    )

    $benchGpuOut = Join-Path $expDir "benchmark_gpu.json"
    $benchCpuOut = Join-Path $expDir "benchmark_cpu.json"
    if (-not $OfflineOnly -and -not $SkipBenchmarks) {
        $benchVideo = "data/demo_videos/sherbrooke_intersection.mp4"
        $benchCalib = "data/demo_videos/calib_sherbrooke_intersection.yaml"
        if (-not (Test-Path $benchVideo) -or -not (Test-Path $benchCalib)) {
            throw "Missing benchmark input files: $benchVideo / $benchCalib"
        }
        Invoke-PyStep -Label "benchmark-gpu" -PyExe $PyExe -PyArgs @(
            "-m", "analysis.benchmark",
            "--source", $benchVideo,
            "--calib", $benchCalib,
            "--device", $Device,
            "--max-frames", "120",
            "--speed-model", "ctrv",
            "--conflict-model", "obb_ssm",
            "--enable-stitching",
            "--include-cyclist-vru",
            "--output", $benchGpuOut
        )
        Invoke-PyStep -Label "benchmark-cpu" -PyExe $PyExe -PyArgs @(
            "-m", "analysis.benchmark",
            "--source", $benchVideo,
            "--calib", $benchCalib,
            "--device", "cpu",
            "--max-frames", "60",
            "--speed-model", "ctrv",
            "--conflict-model", "obb_ssm",
            "--enable-stitching",
            "--include-cyclist-vru",
            "--output", $benchCpuOut
        )
    }

    if (-not $OfflineOnly -and -not $SkipDemos) {
        foreach ($d in $demos) {
            $demoDir = Join-Path $OutputRoot "demo_$($d.Name)"
            $evalOut = Join-Path $expDir "eval_$($d.Name).md"
            Invoke-PyStep -Label "evaluate-$($d.Name)" -PyExe $PyExe -PyArgs @(
                "-m", "analysis.evaluate",
                "--run", $demoDir,
                "--validate", "--markdown",
                "--output", $evalOut
            )
        }
    }

    if (-not $SkipCharts) {
        $chartsDir = Join-Path $expDir "charts"
        $chartArgs = @("-m", "analysis.charts", "--ablation", $ablationOut)
        $sherbrookeRun = Join-Path $OutputRoot "demo_sherbrooke_intersection"
        if (-not $OfflineOnly -and ($DryRun -or (Test-Path (Join-Path $sherbrookeRun "speeds.csv")))) {
            $chartArgs += @("--run", $sherbrookeRun)
        }
        if (-not $OfflineOnly -and ($DryRun -or (Test-Path $benchGpuOut))) {
            $chartArgs += @("--benchmark", $benchGpuOut)
        }
        $chartArgs += @("--output-dir", $chartsDir)
        Invoke-PyStep -Label "charts" -PyExe $PyExe -PyArgs $chartArgs
    }

    Write-Host "[done] Reproduce pipeline finished successfully. Output root: $OutputRoot"
}
finally {
    Pop-Location
}
