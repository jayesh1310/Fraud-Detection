<#
.SYNOPSIS
    Sets JAVA_HOME to JDK 17 and adds it to the system PATH permanently.

.DESCRIPTION
    This script:
      1. Auto-detects JDK 17 from common install locations.
      2. Sets JAVA_HOME as a machine-level environment variable (requires admin).
      3. Prepends JAVA_HOME\bin to the machine-level PATH.
      4. Refreshes the current session so java -version works immediately.
      5. Verifies the installation by running java -version.

.NOTES
    Run this script as Administrator:
        Start-Process powershell -Verb RunAs -ArgumentList "-File set_java_env.ps1"
#>

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Status($icon, $color, $msg) {
    Write-Host "  $icon  $msg" -ForegroundColor $color
}
function OK($msg)   { Write-Status "[OK]"   Green  $msg }
function FAIL($msg) { Write-Status "[FAIL]" Red    $msg }
function INFO($msg) { Write-Host   "         --> $msg" -ForegroundColor Cyan }
function SECTION($title) {
    Write-Host ""
    Write-Host ("=" * 54) -ForegroundColor DarkCyan
    Write-Host "  $title" -ForegroundColor White
    Write-Host ("=" * 54) -ForegroundColor DarkCyan
}

# ---------------------------------------------------------------------------
# 0. Admin check
# ---------------------------------------------------------------------------
SECTION "Administrator Check"
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    FAIL "This script must be run as Administrator to write system environment variables."
    INFO "Right-click PowerShell -> 'Run as administrator', then re-run."
    exit 1
}
OK "Running as Administrator"

# ---------------------------------------------------------------------------
# 1. Locate JDK 17
# ---------------------------------------------------------------------------
SECTION "Locating JDK 17"

$SearchRoots = @(
    "C:\Program Files\Eclipse Adoptium",
    "C:\Program Files\Eclipse Foundation",
    "C:\Program Files\Microsoft",
    "C:\Program Files\Amazon Corretto",
    "C:\Program Files\Java",
    "C:\Program Files\Zulu"
)

$JdkPath = $null

foreach ($root in $SearchRoots) {
    if (Test-Path $root) {
        $candidates = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "jdk[-_]?17" } |
            Sort-Object Name -Descending
        foreach ($c in $candidates) {
            $javaExe = Join-Path $c.FullName "bin\java.exe"
            if (Test-Path $javaExe) {
                $JdkPath = $c.FullName
                break
            }
        }
    }
    if ($JdkPath) { break }
}

if (-not $JdkPath) {
    FAIL "Could not auto-detect a JDK 17 installation."
    INFO "Edit the SearchRoots array in this script or set JdkPath manually."
    exit 1
}

OK "JDK 17 found at: $JdkPath"

# ---------------------------------------------------------------------------
# 2. Set JAVA_HOME (Machine scope)
# ---------------------------------------------------------------------------
SECTION "Setting JAVA_HOME"

$currentJavaHome = [System.Environment]::GetEnvironmentVariable("JAVA_HOME", "Machine")

if ($currentJavaHome -eq $JdkPath) {
    OK "JAVA_HOME is already correctly set -- no change needed."
} else {
    if ($currentJavaHome) {
        Write-Host "  [INFO] Replacing existing JAVA_HOME: $currentJavaHome" -ForegroundColor Yellow
    }
    [System.Environment]::SetEnvironmentVariable("JAVA_HOME", $JdkPath, "Machine")
    OK "JAVA_HOME set to: $JdkPath  (Machine scope)"
}

# ---------------------------------------------------------------------------
# 3. Add JAVA_HOME\bin to system PATH
# ---------------------------------------------------------------------------
SECTION "Updating System PATH"

$javaBin     = Join-Path $JdkPath "bin"
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
$pathEntries = $currentPath -split ";" | Where-Object { $_ -ne "" }

# Remove stale JDK/JRE entries so we don't accumulate duplicates
$cleanedPath = $pathEntries | Where-Object {
    ($_ -notmatch "\\jdk") -and ($_ -notmatch "\\jre") -and ($_ -ne $javaBin)
}

$newPath = ($javaBin + ";" + ($cleanedPath -join ";")).TrimEnd(";")
[System.Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
OK "Added '$javaBin' to the front of system PATH"

# ---------------------------------------------------------------------------
# 4. Refresh current session
# ---------------------------------------------------------------------------
SECTION "Refreshing Current Session"

$env:JAVA_HOME = $JdkPath
$env:PATH      = $javaBin + ";" + $env:PATH
OK "Current session updated -- JAVA_HOME and PATH are active"

# ---------------------------------------------------------------------------
# 5. Verify: java -version
# ---------------------------------------------------------------------------
SECTION "Verification -- java -version"

Write-Host ""
Write-Host "  Running: java -version" -ForegroundColor White
Write-Host ""

try {
    $output = & java -version 2>&1
    $versionText = ($output | Out-String).Trim()
    Write-Host $versionText -ForegroundColor Gray
    Write-Host ""

    if ($versionText -match "17\.") {
        OK "Java 17 confirmed!"
    } elseif ($versionText -match "\d+\.\d+") {
        Write-Host "  [WARN]  Java found but version may not be 17. Check output above." -ForegroundColor Yellow
    } else {
        FAIL "java -version did not return recognisable output."
    }
} catch {
    FAIL "java -version failed: $_"
}

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
SECTION "Summary"

$javaCmd = Get-Command java -ErrorAction SilentlyContinue
$javaCmdPath = if ($javaCmd) { $javaCmd.Source } else { "not found" }

Write-Host ""
Write-Host "  JAVA_HOME : $($env:JAVA_HOME)" -ForegroundColor Green
Write-Host "  java      : $javaCmdPath"       -ForegroundColor Green
Write-Host ""
Write-Host ("=" * 54) -ForegroundColor DarkCyan
Write-Host "  Done! Open a new terminal to pick up system-level changes." -ForegroundColor White
Write-Host ("=" * 54) -ForegroundColor DarkCyan
Write-Host ""
