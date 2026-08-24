#Requires -RunAsAdministrator

[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$HostName = "tx2-measurement.barnstxprod.local",
    [string]$SiteName = "TX2 Measurement",
    [string]$ApplicationPool = "TX2MeasurementPool",
    [string]$PhysicalPath = (Join-Path $PSScriptRoot "iis"),
    [Parameter(Mandatory)]
    [string]$CertificateThumbprint,
    [Parameter(Mandatory)]
    [string[]]$AllowedRemoteAddress,
    [switch]$SkipBackendHealthCheck
)

$ErrorActionPreference = "Stop"
$appcmd = Join-Path $env:WINDIR "System32\inetsrv\appcmd.exe"
if (-not (Test-Path -LiteralPath $appcmd)) {
    throw "IIS is not installed. Install IIS, Windows Authentication, URL Rewrite and ARR first."
}

if (-not $SkipBackendHealthCheck) {
    $health = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri "http://127.0.0.1:8767/api/health" `
        -TimeoutSec 15
    if ($health.StatusCode -ne 200) {
        throw "The Waitress backend is not healthy on 127.0.0.1:8767."
    }
}

Import-Module WebAdministration
$globalModules = @(Get-WebGlobalModule | Select-Object -ExpandProperty Name)
if ($globalModules -notcontains "RewriteModule") {
    throw "IIS URL Rewrite is not installed."
}
if ($globalModules -notcontains "WindowsAuthenticationModule") {
    throw "The IIS Windows Authentication feature is not installed."
}
try {
    Get-WebConfigurationProperty `
        -PSPath "MACHINE/WEBROOT/APPHOST" `
        -Filter "system.webServer/proxy" `
        -Name "enabled" | Out-Null
}
catch {
    throw "Application Request Routing (ARR) is not installed."
}

$thumbprint = $CertificateThumbprint.Replace(" ", "").ToUpperInvariant()
$certificate = Get-Item -LiteralPath "Cert:\LocalMachine\My\$thumbprint"
$certificateName = $certificate.GetNameInfo(
    [Security.Cryptography.X509Certificates.X509NameType]::DnsName,
    $false
)
$certificateMatches = $certificateName -ieq $HostName
if (-not $certificateMatches -and $certificateName.StartsWith("*.")) {
    $certificateMatches = $HostName.EndsWith($certificateName.Substring(1))
}
if (-not $certificateMatches) {
    throw "Certificate DNS name '$certificateName' does not match '$HostName'."
}
if ($certificate.NotAfter -le (Get-Date)) {
    throw "The selected certificate is expired."
}

$resolvedPhysicalPath = [IO.Path]::GetFullPath($PhysicalPath)
if (-not (Test-Path -LiteralPath (Join-Path $resolvedPhysicalPath "web.config"))) {
    throw "The IIS web.config was not found under: $resolvedPhysicalPath"
}

if ($PSCmdlet.ShouldProcess($SiteName, "Configure IIS reverse proxy")) {
    if (-not (Test-Path "IIS:\AppPools\$ApplicationPool")) {
        New-WebAppPool -Name $ApplicationPool | Out-Null
    }
    Set-ItemProperty "IIS:\AppPools\$ApplicationPool" managedRuntimeVersion ""
    Set-ItemProperty "IIS:\AppPools\$ApplicationPool" startMode "AlwaysRunning"
    Set-ItemProperty `
        "IIS:\AppPools\$ApplicationPool" `
        processModel.idleTimeout `
        ([TimeSpan]::Zero)

    $appPoolIdentity = "IIS AppPool\$ApplicationPool"
    & "$env:WINDIR\System32\icacls.exe" `
        $resolvedPhysicalPath `
        /grant "${appPoolIdentity}:(OI)(CI)(RX)" `
        /T `
        /C | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to grant the IIS application pool read access."
    }

    if (-not (Test-Path "IIS:\Sites\$SiteName")) {
        New-Website `
            -Name $SiteName `
            -PhysicalPath $resolvedPhysicalPath `
            -ApplicationPool $ApplicationPool `
            -IPAddress "*" `
            -Port 443 `
            -HostHeader $HostName `
            -Ssl | Out-Null
    }
    else {
        Set-ItemProperty "IIS:\Sites\$SiteName" physicalPath $resolvedPhysicalPath
        Set-ItemProperty "IIS:\Sites\$SiteName" applicationPool $ApplicationPool
        $binding = Get-WebBinding -Name $SiteName -Protocol https |
            Where-Object { $_.bindingInformation -eq "*:443:$HostName" }
        if (-not $binding) {
            New-WebBinding `
                -Name $SiteName `
                -Protocol https `
                -IPAddress "*" `
                -Port 443 `
                -HostHeader $HostName `
                -SslFlags 1
        }
    }

    Set-WebBinding `
        -Name $SiteName `
        -Protocol https `
        -BindingInformation "*:443:$HostName" `
        -PropertyName sslFlags `
        -Value 1

    Set-WebConfigurationProperty `
        -PSPath "IIS:\" `
        -Location $SiteName `
        -Filter "system.webServer/security/authentication/anonymousAuthentication" `
        -Name enabled `
        -Value false
    Set-WebConfigurationProperty `
        -PSPath "IIS:\" `
        -Location $SiteName `
        -Filter "system.webServer/security/authentication/windowsAuthentication" `
        -Name enabled `
        -Value true

    & $appcmd set config `
        /section:system.webServer/proxy `
        /enabled:"True" `
        /preserveHostHeader:"True" `
        /reverseRewriteHostInResponseHeaders:"False" `
        /commit:apphost | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ARR proxy configuration failed with exit code $LASTEXITCODE."
    }

    Push-Location "IIS:\SslBindings"
    try {
        $sslBinding = "0.0.0.0!443!$HostName"
        if (-not (Test-Path -LiteralPath $sslBinding)) {
            $certificate | New-Item $sslBinding -SSLFlags 1 | Out-Null
        }
        else {
            $boundCertificate = Get-Item -LiteralPath $sslBinding
            if ($boundCertificate.Thumbprint -ne $thumbprint) {
                throw "The HTTPS binding already uses a different certificate."
            }
        }
    }
    finally {
        Pop-Location
    }

    $firewallRuleName = "TX2 Measurement HTTPS"
    $firewallRule = Get-NetFirewallRule `
        -DisplayName $firewallRuleName `
        -ErrorAction SilentlyContinue
    if ($firewallRule) {
        $firewallRule | Set-NetFirewallRule `
            -Enabled True `
            -Profile Domain `
            -Direction Inbound `
            -Action Allow
        $firewallRule | Get-NetFirewallAddressFilter |
            Set-NetFirewallAddressFilter -RemoteAddress $AllowedRemoteAddress
    }
    else {
        New-NetFirewallRule `
            -DisplayName $firewallRuleName `
            -Direction Inbound `
            -Action Allow `
            -Enabled True `
            -Profile Domain `
            -Protocol TCP `
            -LocalPort 443 `
            -RemoteAddress $AllowedRemoteAddress | Out-Null
    }

    Start-Website -Name $SiteName
}

Write-Output "IIS site configured: https://$HostName"
Write-Warning "DNS must map $HostName to this server before VPN clients can connect."
