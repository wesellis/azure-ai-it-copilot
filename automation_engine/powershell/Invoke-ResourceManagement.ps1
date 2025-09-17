#Requires -Version 7.0
#Requires -Modules Az

<#
.SYNOPSIS
    Azure resource management automation script
.DESCRIPTION
    Handles creation, modification, and deletion of Azure resources
.PARAMETER Action
    The action to perform (Create, Update, Delete, Query)
.PARAMETER ResourceType
    Type of Azure resource to manage
.PARAMETER Configuration
    JSON configuration for the resource
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Create', 'Update', 'Delete', 'Query')]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [string]$ResourceType,

    [Parameter(Mandatory=$false)]
    [string]$Configuration,

    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup,

    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId,

    [switch]$DryRun
)

# Import required modules
Import-Module Az.Accounts -ErrorAction Stop
Import-Module Az.Resources -ErrorAction Stop
Import-Module Az.Compute -ErrorAction Stop
Import-Module Az.Network -ErrorAction Stop
Import-Module Az.Storage -ErrorAction Stop

# Initialize logging
$LogPath = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath | Out-Null
}

$LogFile = Join-Path $LogPath "resource-management-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level = 'Info'
    )

    $Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $LogMessage = "[$Timestamp] [$Level] $Message"

    Add-Content -Path $LogFile -Value $LogMessage

    switch ($Level) {
        'Warning' { Write-Warning $Message }
        'Error' { Write-Error $Message }
        default { Write-Host $Message }
    }
}

function Connect-AzureEnvironment {
    <#
    .SYNOPSIS
        Connects to Azure environment
    #>
    param(
        [string]$SubscriptionId
    )

    try {
        $Context = Get-AzContext

        if (-not $Context) {
            Write-Log "Connecting to Azure..." -Level Info
            Connect-AzAccount -ErrorAction Stop
        }

        if ($SubscriptionId) {
            Write-Log "Setting subscription: $SubscriptionId" -Level Info
            Set-AzContext -SubscriptionId $SubscriptionId -ErrorAction Stop
        }

        $Context = Get-AzContext
        Write-Log "Connected to Azure Subscription: $($Context.Subscription.Name)" -Level Info

        return $true
    }
    catch {
        Write-Log "Failed to connect to Azure: $_" -Level Error
        return $false
    }
}

function New-AzureResource {
    <#
    .SYNOPSIS
        Creates a new Azure resource
    #>
    param(
        [string]$ResourceType,
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    Write-Log "Creating new $ResourceType resource" -Level Info

    try {
        switch ($ResourceType.ToLower()) {
            'virtualmachine' {
                $Result = New-VirtualMachine -Config $Config -ResourceGroup $ResourceGroup
            }
            'storageaccount' {
                $Result = New-StorageAccount -Config $Config -ResourceGroup $ResourceGroup
            }
            'webapp' {
                $Result = New-WebApp -Config $Config -ResourceGroup $ResourceGroup
            }
            'sqldatabase' {
                $Result = New-SqlDatabase -Config $Config -ResourceGroup $ResourceGroup
            }
            'keyvault' {
                $Result = New-KeyVault -Config $Config -ResourceGroup $ResourceGroup
            }
            default {
                throw "Unsupported resource type: $ResourceType"
            }
        }

        Write-Log "Successfully created $ResourceType resource" -Level Info
        return $Result
    }
    catch {
        Write-Log "Failed to create $ResourceType: $_" -Level Error
        throw
    }
}

function New-VirtualMachine {
    param(
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    # Ensure resource group exists
    $RG = Get-AzResourceGroup -Name $ResourceGroup -ErrorAction SilentlyContinue
    if (-not $RG) {
        Write-Log "Creating resource group: $ResourceGroup" -Level Info
        $Location = $Config.Location ?? 'eastus'
        New-AzResourceGroup -Name $ResourceGroup -Location $Location
    }

    # Create VM configuration
    $VMConfig = @{
        ResourceGroupName = $ResourceGroup
        Name = $Config.Name
        Location = $Config.Location ?? 'eastus'
        Size = $Config.Size ?? 'Standard_B2s'
    }

    if ($DryRun) {
        Write-Log "DRY RUN: Would create VM with configuration:" -Level Info
        $VMConfig | ConvertTo-Json | Write-Log
        return $VMConfig
    }

    # Create network interface if not exists
    $NicName = "$($Config.Name)-nic"
    $Nic = Get-AzNetworkInterface -Name $NicName -ResourceGroupName $ResourceGroup -ErrorAction SilentlyContinue

    if (-not $Nic) {
        # Create virtual network if needed
        $VNetName = "$ResourceGroup-vnet"
        $VNet = Get-AzVirtualNetwork -Name $VNetName -ResourceGroupName $ResourceGroup -ErrorAction SilentlyContinue

        if (-not $VNet) {
            $SubnetConfig = New-AzVirtualNetworkSubnetConfig -Name "default" -AddressPrefix "10.0.0.0/24"
            $VNet = New-AzVirtualNetwork -ResourceGroupName $ResourceGroup -Location $VMConfig.Location `
                -Name $VNetName -AddressPrefix "10.0.0.0/16" -Subnet $SubnetConfig
        }

        # Create public IP
        $PublicIP = New-AzPublicIpAddress -Name "$($Config.Name)-ip" -ResourceGroupName $ResourceGroup `
            -Location $VMConfig.Location -AllocationMethod Dynamic

        # Create network interface
        $Nic = New-AzNetworkInterface -Name $NicName -ResourceGroupName $ResourceGroup `
            -Location $VMConfig.Location -SubnetId $VNet.Subnets[0].Id -PublicIpAddressId $PublicIP.Id
    }

    # Create VM
    $VMCredential = New-Object System.Management.Automation.PSCredential ("azureuser", (ConvertTo-SecureString "P@ssw0rd123!" -AsPlainText -Force))

    $VM = New-AzVMConfig -VMName $Config.Name -VMSize $VMConfig.Size
    $VM = Set-AzVMOperatingSystem -VM $VM -Linux -ComputerName $Config.Name -Credential $VMCredential
    $VM = Set-AzVMSourceImage -VM $VM -PublisherName "Canonical" -Offer "UbuntuServer" -Skus "18.04-LTS" -Version "latest"
    $VM = Add-AzVMNetworkInterface -VM $VM -Id $Nic.Id
    $VM = Set-AzVMBootDiagnostic -VM $VM -Disable

    $Result = New-AzVM -ResourceGroupName $ResourceGroup -Location $VMConfig.Location -VM $VM

    return $Result
}

function New-StorageAccount {
    param(
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    $StorageConfig = @{
        ResourceGroupName = $ResourceGroup
        Name = $Config.Name -replace '[^a-z0-9]', '' # Storage account names must be lowercase and alphanumeric
        Location = $Config.Location ?? 'eastus'
        SkuName = $Config.Sku ?? 'Standard_LRS'
        Kind = $Config.Kind ?? 'StorageV2'
    }

    if ($DryRun) {
        Write-Log "DRY RUN: Would create storage account with configuration:" -Level Info
        $StorageConfig | ConvertTo-Json | Write-Log
        return $StorageConfig
    }

    return New-AzStorageAccount @StorageConfig
}

function Update-AzureResource {
    param(
        [string]$ResourceType,
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    Write-Log "Updating $ResourceType resource" -Level Info

    try {
        switch ($ResourceType.ToLower()) {
            'virtualmachine' {
                # Update VM size
                if ($Config.NewSize) {
                    $VM = Get-AzVM -ResourceGroupName $ResourceGroup -Name $Config.Name
                    $VM.HardwareProfile.VmSize = $Config.NewSize

                    if (-not $DryRun) {
                        Update-AzVM -VM $VM -ResourceGroupName $ResourceGroup
                    }
                }
            }
            'storageaccount' {
                # Update storage account properties
                if ($Config.EnableHttpsOnly) {
                    Set-AzStorageAccount -ResourceGroupName $ResourceGroup -Name $Config.Name `
                        -EnableHttpsTrafficOnly $true
                }
            }
            default {
                throw "Update not implemented for resource type: $ResourceType"
            }
        }

        Write-Log "Successfully updated $ResourceType resource" -Level Info
    }
    catch {
        Write-Log "Failed to update $ResourceType: $_" -Level Error
        throw
    }
}

function Remove-AzureResource {
    param(
        [string]$ResourceType,
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    Write-Log "Removing $ResourceType resource: $($Config.Name)" -Level Warning

    if ($DryRun) {
        Write-Log "DRY RUN: Would remove $ResourceType: $($Config.Name)" -Level Info
        return @{ Status = "DryRun"; Message = "Resource would be deleted" }
    }

    try {
        switch ($ResourceType.ToLower()) {
            'virtualmachine' {
                Remove-AzVM -ResourceGroupName $ResourceGroup -Name $Config.Name -Force
            }
            'storageaccount' {
                Remove-AzStorageAccount -ResourceGroupName $ResourceGroup -Name $Config.Name -Force
            }
            'resourcegroup' {
                Remove-AzResourceGroup -Name $Config.Name -Force
            }
            default {
                throw "Delete not implemented for resource type: $ResourceType"
            }
        }

        Write-Log "Successfully removed $ResourceType resource" -Level Info
    }
    catch {
        Write-Log "Failed to remove $ResourceType: $_" -Level Error
        throw
    }
}

function Get-AzureResourceInfo {
    param(
        [string]$ResourceType,
        [hashtable]$Config,
        [string]$ResourceGroup
    )

    Write-Log "Querying $ResourceType resources" -Level Info

    try {
        $Results = switch ($ResourceType.ToLower()) {
            'virtualmachine' {
                if ($Config.Name) {
                    Get-AzVM -ResourceGroupName $ResourceGroup -Name $Config.Name
                } else {
                    Get-AzVM -ResourceGroupName $ResourceGroup
                }
            }
            'storageaccount' {
                if ($Config.Name) {
                    Get-AzStorageAccount -ResourceGroupName $ResourceGroup -Name $Config.Name
                } else {
                    Get-AzStorageAccount -ResourceGroupName $ResourceGroup
                }
            }
            'all' {
                Get-AzResource -ResourceGroupName $ResourceGroup
            }
            default {
                Get-AzResource -ResourceType $ResourceType -ResourceGroupName $ResourceGroup
            }
        }

        return $Results
    }
    catch {
        Write-Log "Failed to query $ResourceType: $_" -Level Error
        throw
    }
}

# Main execution
try {
    Write-Log "Starting resource management operation: $Action" -Level Info

    # Connect to Azure
    if (-not (Connect-AzureEnvironment -SubscriptionId $SubscriptionId)) {
        throw "Failed to connect to Azure"
    }

    # Parse configuration
    $Config = if ($Configuration) {
        $Configuration | ConvertFrom-Json -AsHashtable
    } else {
        @{}
    }

    # Execute action
    $Result = switch ($Action) {
        'Create' {
            New-AzureResource -ResourceType $ResourceType -Config $Config -ResourceGroup $ResourceGroup
        }
        'Update' {
            Update-AzureResource -ResourceType $ResourceType -Config $Config -ResourceGroup $ResourceGroup
        }
        'Delete' {
            Remove-AzureResource -ResourceType $ResourceType -Config $Config -ResourceGroup $ResourceGroup
        }
        'Query' {
            Get-AzureResourceInfo -ResourceType $ResourceType -Config $Config -ResourceGroup $ResourceGroup
        }
    }

    # Output results
    $Output = @{
        Status = 'Success'
        Action = $Action
        ResourceType = $ResourceType
        Result = $Result
        Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    }

    $Output | ConvertTo-Json -Depth 10
    Write-Log "Operation completed successfully" -Level Info
}
catch {
    $ErrorOutput = @{
        Status = 'Error'
        Action = $Action
        ResourceType = $ResourceType
        Error = $_.Exception.Message
        Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    }

    $ErrorOutput | ConvertTo-Json -Depth 10
    Write-Log "Operation failed: $_" -Level Error
    exit 1
}