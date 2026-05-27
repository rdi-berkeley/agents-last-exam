param(
    [string]$ProjectPath = '',
    [string]$SubmissionProject = '',
    [string]$MapPathOverride = '',
    [string]$OutputDir = '',
    [string]$EngineCmd = 'C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor-Cmd.exe',
    [string]$TempRenderDir = '',
    [int]$MaxRetries = 2,
    [string]$RenderConfigPath = '',
    [string]$ReportPath = ''
)

$ErrorActionPreference = 'Stop'

if (-not $RenderConfigPath) {
    throw 'RenderConfigPath is required.'
}

$renderConfig = Get-Content -LiteralPath $RenderConfigPath -Raw | ConvertFrom-Json

if (-not $ProjectPath) {
    if ($SubmissionProject) {
        $ProjectPath = $SubmissionProject
    } elseif ($MapPathOverride) {
        throw 'ProjectPath or SubmissionProject is required when MapPathOverride is used.'
    } elseif ($renderConfig.default_submission_project_path) {
        throw 'ProjectPath or SubmissionProject must be provided explicitly for candidate rendering.'
    }
}

if (-not $SubmissionProject -and -not $MapPathOverride) {
    $SubmissionProject = $ProjectPath
}

if (-not $OutputDir) {
    throw 'OutputDir is required.'
}

if (-not $TempRenderDir) {
    $TempRenderDir = Join-Path ([System.IO.Path]::GetTempPath()) 'agenthle_unreal_scene_restoration_renders'
}

$targetMapPath = if ($MapPathOverride) { $MapPathOverride } else { $renderConfig.submission_map_path }
$configAssetPath = "$($renderConfig.config_asset_path)"
$configAssetName = "$($renderConfig.config_asset_name)"
if (-not $configAssetName -and $configAssetPath) {
    $segments = ($configAssetPath -replace '\\', '/') -split '/'
    $configAssetName = $segments[-1]
}
$config = if ($configAssetPath -and $configAssetName) {
    "$configAssetPath.$configAssetName"
} else {
    "$configAssetPath"
}
if ($config.EndsWith('.') -and $configAssetPath) {
    $segments = ($configAssetPath -replace '\\', '/') -split '/'
    $leafName = $segments[-1]
    if ($leafName) {
        $config = "$configAssetPath.$leafName"
    }
}
$sequences = if ($renderConfig.sequence_names) { @($renderConfig.sequence_names) } elseif ($renderConfig.sequences) { @($renderConfig.sequences) } else { @() }
if ($sequences.Count -eq 0) {
    throw 'render_config.json does not define any sequence names.'
}
$null = New-Item -ItemType Directory -Force -Path $OutputDir
$null = New-Item -ItemType Directory -Force -Path $TempRenderDir

if (-not (Test-Path -LiteralPath $ProjectPath)) {
    throw "Project file not found: $ProjectPath"
}

$projectRoot = Split-Path -Parent $ProjectPath
$projectPythonDir = Join-Path $projectRoot 'Content\Python'
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$runtimeExecutorSource = Join-Path $scriptRoot 'runtime_executor.py'
$runtimeInitSource = Join-Path $scriptRoot 'runtime_init_unreal.py'
$sequenceBuilderSource = Join-Path $scriptRoot 'build_camera_sequences.py'
$runtimeExecutorDest = Join-Path $projectPythonDir 'runtime_executor.py'
$runtimeInitDest = Join-Path $projectPythonDir 'init_unreal.py'
$sequenceBuilderDest = Join-Path $projectPythonDir 'build_camera_sequences.py'
$sequenceBuilderConfigDest = Join-Path $projectPythonDir 'scene_restoration_builder_config.json'
$cameraManifestDest = Join-Path $projectPythonDir 'scene_restoration_camera_manifest.json'
$engineLogDir = Join-Path $TempRenderDir 'engine_logs'
$developersDir = Join-Path $projectRoot 'Content\Developers'

$null = New-Item -ItemType Directory -Force -Path $projectPythonDir
$null = New-Item -ItemType Directory -Force -Path $engineLogDir
Copy-Item -LiteralPath $runtimeExecutorSource -Destination $runtimeExecutorDest -Force
Copy-Item -LiteralPath $runtimeInitSource -Destination $runtimeInitDest -Force
Copy-Item -LiteralPath $sequenceBuilderSource -Destination $sequenceBuilderDest -Force

if (Test-Path -LiteralPath $developersDir) {
    Write-Host "Removing eval-copy Developers content: $developersDir"
    $null = & cmd.exe /c "rmdir /s /q `"$developersDir`""
    if (Test-Path -LiteralPath $developersDir) {
        throw "Failed to remove eval-copy Developers directory: $developersDir"
    }
}

function Resolve-TaskRelativePath {
    param(
        [string]$RenderConfigPath,
        [string]$TaskRelativePath
    )

    if (-not $TaskRelativePath) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($TaskRelativePath)) {
        return $TaskRelativePath
    }

    $renderConfigDir = Split-Path -Parent $RenderConfigPath
    $referenceDir = Split-Path -Parent $renderConfigDir
    $taskRoot = Split-Path -Parent $referenceDir
    $segments = ($TaskRelativePath -replace '/', '\') -split '\\'
    return Join-Path $taskRoot ($segments -join '\')
}

function Convert-DiskPathToAssetPath {
    param(
        [string]$ProjectRoot,
        [string]$DiskPath
    )

    $contentRoot = Join-Path $ProjectRoot 'Content'
    $normalizedContentRoot = [System.IO.Path]::GetFullPath($contentRoot).TrimEnd('\')
    $normalizedDiskPath = [System.IO.Path]::GetFullPath($DiskPath)
    if ($normalizedDiskPath.StartsWith($normalizedContentRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relativePath = $normalizedDiskPath.Substring($normalizedContentRoot.Length).TrimStart('\')
    } else {
        throw "Disk path is not under project Content/: $DiskPath"
    }
    $relativePath = $relativePath -replace '\\', '/'
    $assetPath = "/Game/$relativePath"
    return [System.IO.Path]::ChangeExtension($assetPath, $null) -replace '\.$', ''
}

function Resolve-SequenceRef {
    param(
        [string]$ProjectRoot,
        [string]$ConfiguredFolder,
        [string]$Sequence
    )

    if ($ConfiguredFolder) {
        $configuredRelative = ($ConfiguredFolder -replace '^/Game/', '') -replace '/', '\'
        $configuredDiskPath = Join-Path (Join-Path $ProjectRoot 'Content') ($configuredRelative + '.uasset')
        $configuredDiskPath = Join-Path (Split-Path -Parent $configuredDiskPath) "$Sequence.uasset"
        if (Test-Path -LiteralPath $configuredDiskPath) {
            return "$ConfiguredFolder/$Sequence.$Sequence"
        }
    }

    $match = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'Content') -Recurse -File -Filter "$Sequence.uasset" |
        Select-Object -First 1
    if (-not $match) {
        throw "Unable to locate level sequence asset for $Sequence under $ProjectRoot\\Content"
    }

    $resolvedFolder = Split-Path -Parent $match.FullName
    $resolvedFolderAssetPath = Convert-DiskPathToAssetPath -ProjectRoot $ProjectRoot -DiskPath $resolvedFolder
    return "$resolvedFolderAssetPath/$Sequence.$Sequence"
}

function Resolve-ConfigRef {
    param(
        [string]$ProjectRoot,
        [string]$ConfiguredAssetPath,
        [string]$ConfiguredAssetName
    )

    if ($ConfiguredAssetPath) {
        $configuredRelative = ($ConfiguredAssetPath -replace '^/Game/', '') -replace '/', '\'
        $configuredDiskPath = Join-Path (Join-Path $ProjectRoot 'Content') ($configuredRelative + '.uasset')
        if (Test-Path -LiteralPath $configuredDiskPath) {
            if ($ConfiguredAssetName) {
                return "$ConfiguredAssetPath.$ConfiguredAssetName"
            }
            $leaf = Split-Path -Leaf $ConfiguredAssetPath
            return "$ConfiguredAssetPath.$leaf"
        }
    }

    if ($ConfiguredAssetName) {
        $match = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'Content') -Recurse -File -Filter "$ConfiguredAssetName.uasset" |
            Select-Object -First 1
        if ($match) {
            $assetPath = Convert-DiskPathToAssetPath -ProjectRoot $ProjectRoot -DiskPath $match.FullName
            $leaf = Split-Path -Leaf $assetPath
            return "$assetPath.$leaf"
        }
    }

    return $null
}

function Stop-TaskRenderProcesses {
    param(
        [string]$ProjectPath
    )

    if (-not $ProjectPath) {
        return
    }

    $normalizedProjectPath = $ProjectPath.Replace('/', '\').ToLowerInvariant()
    $staleProcesses = Get-CimInstance Win32_Process -Filter "Name = 'UnrealEditor-Cmd.exe'" |
        Where-Object {
            $_.CommandLine -and $_.CommandLine.ToLowerInvariant().Contains($normalizedProjectPath)
        }
    foreach ($proc in $staleProcesses) {
        Write-Host "Stopping stale UnrealEditor-Cmd PID=$($proc.ProcessId) for $ProjectPath"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Write-SequenceFailureSummary {
    param(
        [string]$Sequence,
        [int]$Attempt,
        [string]$Reason,
        [string]$RenderLogPath,
        [string]$TempFile,
        [string]$OutFile
    )

    $summaryDir = Join-Path $engineLogDir 'failure_summaries'
    $null = New-Item -ItemType Directory -Force -Path $summaryDir
    $summaryPath = Join-Path $summaryDir "$Sequence.attempt$Attempt.summary.txt"
    $summary = @(
        "sequence=$Sequence"
        "attempt=$Attempt"
        "reason=$Reason"
        "render_log_path=$RenderLogPath"
        "temp_file=$TempFile"
        "out_file=$OutFile"
    )
    if (Test-Path -LiteralPath $RenderLogPath) {
        $summary += ""
        $summary += "== render log tail =="
        $summary += @(Get-Content -LiteralPath $RenderLogPath -Tail 120 -ErrorAction SilentlyContinue)
    }
    $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-Host "Wrote render failure summary: $summaryPath"
}

function Build-CameraSequences {
    param(
        [string]$ProjectPath,
        [string]$MapPath,
        [string]$CameraManifestPath,
        [string]$SequenceRoot
    )

    if (-not (Test-Path -LiteralPath $CameraManifestPath)) {
        throw "Camera manifest not found: $CameraManifestPath"
    }
    Copy-Item -LiteralPath $CameraManifestPath -Destination $cameraManifestDest -Force

    $builderScript = $sequenceBuilderDest
    if (-not (Test-Path -LiteralPath $builderScript)) {
        throw "Sequence builder script not found: $builderScript"
    }

    Write-Host "Building temporary camera sequences from $CameraManifestPath"
    Stop-TaskRenderProcesses -ProjectPath $ProjectPath
    $builderLogPath = Join-Path $engineLogDir 'build_camera_sequences.log'
    $builderDebugPath = Join-Path $engineLogDir 'build_camera_sequences.debug.txt'
    if (Test-Path -LiteralPath $builderLogPath) {
        Remove-Item -LiteralPath $builderLogPath -Force
    }
    if (Test-Path -LiteralPath $builderDebugPath) {
        Remove-Item -LiteralPath $builderDebugPath -Force
    }
    $builderConfig = [pscustomobject]@{
        camera_manifest_path = $cameraManifestDest
        sequence_root = $SequenceRoot
        debug_log_path = $builderDebugPath
    }
    $builderConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $sequenceBuilderConfigDest -Encoding utf8

    $builderArgs = @(
        $ProjectPath,
        '-run=pythonscript',
        "-script=$builderScript",
        '-stdout',
        '-FullStdOutLogOutput',
        '-Unattended',
        '-NoSplash',
        '-NoSound',
        '-NoP4'
    )
    & $EngineCmd @builderArgs *>&1 | Tee-Object -FilePath $builderLogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Building camera sequences failed with exit code $LASTEXITCODE. See $builderLogPath"
    }
}

function Invoke-SequenceRender {
    param(
        [string]$Sequence,
        [string]$SequenceRef,
        [string]$TempFile,
        [string]$OutFile
    )

    for ($attempt = 1; $attempt -le ($MaxRetries + 1); $attempt++) {
        if (Test-Path -LiteralPath $TempFile) {
            Remove-Item -LiteralPath $TempFile -Force
        }

        Write-Host "Rendering $Sequence (attempt $attempt/$($MaxRetries + 1))"
        Stop-TaskRenderProcesses -ProjectPath $ProjectPath
        $renderLogPath = Join-Path $engineLogDir "$Sequence.attempt$attempt.log"
        if (Test-Path -LiteralPath $renderLogPath) {
            Remove-Item -LiteralPath $renderLogPath -Force
        }
        $engineArgs = @(
            $ProjectPath,
            $targetMapPath,
            '-game',
            '-MoviePipelineLocalExecutorClass=/Script/MovieRenderPipelineCore.MoviePipelinePythonHostExecutor',
            '-ExecutorPythonClass=/Engine/PythonTypes.SceneRestorationRuntimeExecutor',
            "-LevelSequence=$SequenceRef",
            "-SceneRestorationOutputDir=$OutputDir",
            '-windowed',
            "-resx=$($renderConfig.output_resolution[0])",
            "-resy=$($renderConfig.output_resolution[1])",
            '-NoTextureStreaming',
            '-stdout',
            '-FullStdOutLogOutput',
            '-Unattended',
            '-NoSplash',
            '-NoSound',
            '-NoP4'
        )
        if ($script:ResolvedConfigRef) {
            $engineArgs += "-MoviePipelineConfig=$script:ResolvedConfigRef"
        }

        & $EngineCmd @engineArgs *>&1 | Tee-Object -FilePath $renderLogPath
        if ($LASTEXITCODE -ne 0) {
            Write-SequenceFailureSummary `
                -Sequence $Sequence `
                -Attempt $attempt `
                -Reason "render-exit-code-$LASTEXITCODE" `
                -RenderLogPath $renderLogPath `
                -TempFile $TempFile `
                -OutFile $OutFile
            throw "Render command failed for $Sequence with exit code $LASTEXITCODE. See $renderLogPath"
        }

        if ((Test-Path -LiteralPath $TempFile) -and ((Get-Item -LiteralPath $TempFile).Length -gt 0)) {
            Copy-Item -LiteralPath $TempFile -Destination $OutFile -Force
            return
        }

        if ((Test-Path -LiteralPath $OutFile) -and ((Get-Item -LiteralPath $OutFile).Length -gt 0)) {
            return
        }

        if ($attempt -le $MaxRetries) {
            Write-Warning "Render missing for $Sequence on attempt $attempt. Retrying."
            Write-SequenceFailureSummary `
                -Sequence $Sequence `
                -Attempt $attempt `
                -Reason "missing-render-output" `
                -RenderLogPath $renderLogPath `
                -TempFile $TempFile `
                -OutFile $OutFile
            Start-Sleep -Seconds 2
            continue
        }

        throw "Expected render was not produced after $($MaxRetries + 1) attempts: $TempFile"
    }
}

$cameraManifestDiskPath = Resolve-TaskRelativePath -RenderConfigPath $RenderConfigPath -TaskRelativePath "$($renderConfig.camera_manifest_path)"
$script:ResolvedConfigRef = Resolve-ConfigRef -ProjectRoot $projectRoot -ConfiguredAssetPath $configAssetPath -ConfiguredAssetName $configAssetName
$script:SequenceRootOverride = $null
if ($cameraManifestDiskPath) {
    $script:SequenceRootOverride = '/Game/__AgenthleSceneRestorationRuntime'
    $script:ResolvedConfigRef = $null
    Build-CameraSequences `
        -ProjectPath $ProjectPath `
        -MapPath $targetMapPath `
        -CameraManifestPath $cameraManifestDiskPath `
        -SequenceRoot $script:SequenceRootOverride
}

foreach ($sequence in $sequences) {
    if ($script:SequenceRootOverride) {
        $sequenceRef = "$($script:SequenceRootOverride)/$sequence.$sequence"
    } else {
        $sequenceRef = Resolve-SequenceRef -ProjectRoot $projectRoot -ConfiguredFolder "$($renderConfig.sequence_folder)" -Sequence $sequence
    }
    $tempFile = Join-Path $TempRenderDir "$sequence.png"
    $outFile = Join-Path $OutputDir "$sequence.png"
    Invoke-SequenceRender -Sequence $sequence -SequenceRef $sequenceRef -TempFile $tempFile -OutFile $OutFile
}

$report = [pscustomobject]@{
    project_path = (Resolve-Path -LiteralPath $ProjectPath).Path
    target_map_path = $targetMapPath
    output_dir = (Resolve-Path -LiteralPath $OutputDir).Path
    temp_render_dir = (Resolve-Path -LiteralPath $TempRenderDir).Path
    render_config_path = (Resolve-Path -LiteralPath $RenderConfigPath).Path
    engine_log_dir = (Resolve-Path -LiteralPath $engineLogDir).Path
    image_count = $sequences.Count
    rendered_images = @($sequences | ForEach-Object { "$_.png" })
}

if ($ReportPath) {
    $outParent = Split-Path -Parent $ReportPath
    if ($outParent) {
        $null = New-Item -ItemType Directory -Force -Path $outParent
    }
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReportPath -Encoding utf8
}

Write-Host "Rendered $($sequences.Count) views to $OutputDir"
