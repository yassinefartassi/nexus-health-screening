# Starts the NEXUS_PFE FL server on this Windows machine.
# Usage: scripts\start_server.ps1 [-Model dermatology|ocular] [-Rounds 5] [-MinClients 2] [-Port 8080]
#
# One server process federates ONE module at a time -- run this script
# twice (different -Port) to run dermatology and ocular concurrently.
#
# See docs\fl_setup.md for the full runbook, including the Windows Firewall
# rule you need first or clients will just hang with no useful error.
param(
    [string[]]$Model = @("dermatology", "ocular"),
    [int]$Rounds = 5,
    [int]$MinClients = 2,
    [int]$Port = 8080,
    [string]$Checkpoint = "",
    [switch]$AllowRandomInit
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

foreach ($m in $Model) {
    $argsList = @("-m", "fl_server.server", "--model", $m, "--rounds", $Rounds, "--min-clients", $MinClients, "--port", $Port)
    if ($Checkpoint -ne "") {
        $argsList += @("--checkpoint", $Checkpoint)
    }
    if ($AllowRandomInit) {
        $argsList += "--allow-random-init"
    }

    Write-Host "Starting FL server ($m) on 0.0.0.0:$Port for $Rounds round(s), waiting for $MinClients client(s)..."
    python @argsList
}
