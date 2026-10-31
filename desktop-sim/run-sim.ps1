# Launches the AprilTag dashboard simulator on http://localhost:8080 (no robot needed).
#
#   powershell -File run-sim.ps1              # synthetic animated targets
#   powershell -File run-sim.ps1 -Webcam      # ...and opens your laptop webcam as the camera
#
# Ctrl+C stops it. Requires the org.json jar in libs/ (downloaded automatically once).
param([switch]$Webcam)
$ErrorActionPreference = "Stop"

$sim = Split-Path -Parent $MyInvocation.MyCommand.Path
$team = Join-Path (Split-Path -Parent $sim) "TeamCode\src\main\java"
$libs = Join-Path $sim "libs"
$out = Join-Path $sim "build\out"

$jbr = "C:\Program Files\Android\Android Studio\jbr\bin"
if (Test-Path "$jbr\javac.exe") { $javac = "$jbr\javac.exe"; $java = "$jbr\java.exe" }
else { $javac = "javac"; $java = "java" }

$json = Join-Path $libs "json.jar"
if (-not (Test-Path $json)) {
    New-Item -ItemType Directory -Force -Path $libs | Out-Null
    Invoke-WebRequest "https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar" `
        -OutFile $json -UseBasicParsing
}

New-Item -ItemType Directory -Force -Path $out | Out-Null
& $javac --release 11 -encoding UTF-8 -Xlint:-options -cp $json -d $out `
    "$team\org\firstinspires\ftc\teamcode\AprilTagConfig.java" `
    "$team\org\firstinspires\ftc\teamcode\AprilTagDashboardServer.java" `
    (Join-Path $sim "src\main\java\org\firstinspires\ftc\teamcode\DesktopSimMain.java")
if ($LASTEXITCODE -ne 0) { throw "compile failed" }

if ($Webcam) {
    Start-Process python -ArgumentList "`"$sim\webcam_tags.py`"" -WorkingDirectory $sim
}
& $java -cp "$out;$json" org.firstinspires.ftc.teamcode.DesktopSimMain
