param(
    [Parameter(Mandatory)][string]$BasePythonRoot,
    [Parameter(Mandatory)][string]$EnvironmentRoot,
    [Parameter(Mandatory)][string]$Output
)
# Workaround for the Conda-based Windows host when its venv launcher cannot initialize PyTorch.
$ErrorActionPreference = 'Stop'
$base = (Resolve-Path -LiteralPath $BasePythonRoot).Path
$venv = (Resolve-Path -LiteralPath $EnvironmentRoot).Path
$target = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $target) {
    throw 'Use a new host directory'
}
$packages = Join-Path $venv 'Lib/site-packages'
if (-not (Test-Path -LiteralPath $packages -PathType Container)) {
    throw 'Install the locked packages in the virtual environment first'
}
$clock = & (Join-Path $base 'python.exe') -I -B -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0 -or $clock -cne '3.10.13') {
    throw 'The qualified Conda host uses CPython 3.10.13'
}
$files = @('python.exe', 'python3.dll', 'python310.dll') | ForEach-Object { Join-Path $base $_ }
$names = @('ffi.dll', 'LIBBZ2.dll', 'libcrypto-3-x64.dll', 'liblzma.dll', 'libssl-3-x64.dll', 'sqlite3.dll', 'zlib.dll')
$files += $names | ForEach-Object { Join-Path $base "Library/bin/$_" }
foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing standard host file $([IO.Path]::GetFileName($file))"
    }
}
New-Item -Path $target -ItemType Directory | Out-Null
foreach ($file in $files) {
    Copy-Item -LiteralPath $file -Destination $target
}
@((Join-Path $base 'Lib'), (Join-Path $base 'DLLs'), $packages) |
    Set-Content -LiteralPath (Join-Path $target 'python310._pth') -Encoding utf8NoBOM
& (Join-Path $target 'python.exe') -I -B -X utf8 -c "import sys, torch, rfdetr, shapely; from pathlib import Path; assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()) for m in (torch, rfdetr, shapely)); print(torch.__version__, shapely.__version__)" $packages
exit $LASTEXITCODE
