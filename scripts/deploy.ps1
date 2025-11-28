# PowerShell Deployment script for AI Avatar Service
param(
    [string]$Environment = "development",
    [switch]$Production,
    [switch]$Cpu,
    [switch]$Foreground,
    [switch]$Down,
    [switch]$Logs,
    [switch]$Status,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Set script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# 색상 출력을 위한 함수
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# 도움말 표시
if ($Help) {
    Write-Output "Usage: .\scripts\deploy.ps1 [options]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Environment ENV    Deployment environment (development|production)"
    Write-Output "  -Production         Shortcut for -Environment production"
    Write-Output "  -Cpu                Use CPU-only configuration"
    Write-Output "  -Foreground         Run in foreground (don't detach)"
    Write-Output "  -Down               Stop all services"
    Write-Output "  -Logs               Follow service logs"
    Write-Output "  -Status             Show service status"
    Write-Output "  -Help               Show this help"
    exit 0
}

# 기본값 설정
if ($Production) {
    $Environment = "production"
}

$ComposeFile = if ($Cpu) { "docker-compose.cpu.yml" } else { "docker-compose.yml" }
$Detach = if (-not $Foreground) { "-d" } else { "" }

Set-Location $ProjectDir

# Initialize Docker Compose version variable
$UseDockerComposeV2 = $false

# Stop services
if ($Down) {
    Write-ColorOutput Yellow "Stopping services..."
    # Check Docker Compose version first
    try {
        docker-compose --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            docker-compose -f $ComposeFile down
        } else {
            throw
        }
    } catch {
        docker compose -f $ComposeFile down
    }
    Write-ColorOutput Green "Services stopped"
    exit 0
}

# View logs
if ($Logs) {
    try {
        docker-compose --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            docker-compose -f $ComposeFile logs -f
        } else {
            throw
        }
    } catch {
        docker compose -f $ComposeFile logs -f
    }
    exit 0
}

# Check status
if ($Status) {
    try {
        docker-compose --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            docker-compose -f $ComposeFile ps
        } else {
            throw
        }
    } catch {
        docker compose -f $ComposeFile ps
    }
    exit 0
}

Write-ColorOutput Green "AI Avatar Service Deployment"
Write-Output "=============================="
Write-Output ""

# Check prerequisites
Write-ColorOutput Blue "Checking prerequisites..."

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Output "  Docker: $dockerVersion"
} catch {
    Write-ColorOutput Red "Error: Docker is not installed"
    Write-Output "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check Docker Compose
$UseDockerComposeV2 = $false
try {
    $composeVersion = docker-compose --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Output "  Docker Compose: $composeVersion"
    } else {
        throw
    }
} catch {
    # Try docker-compose v2 (docker compose)
    try {
        $composeVersion = docker compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Output "  Docker Compose: $composeVersion"
            $UseDockerComposeV2 = $true
        } else {
            throw
        }
    } catch {
        Write-ColorOutput Red "Error: Docker Compose is not installed"
        exit 1
    }
}

# Check NVIDIA GPU (for GPU deployment)
if (-not $Cpu) {
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput Green "  NVIDIA GPU: Available"
            $gpuInfo | ForEach-Object { Write-Output "    $_" }
        } else {
            Write-ColorOutput Yellow "  NVIDIA GPU: Not detected (consider using -Cpu flag)"
        }
    } catch {
        Write-ColorOutput Yellow "  NVIDIA GPU: Not detected (consider using -Cpu flag)"
    }
}

# Check and create .env file
if (-not (Test-Path ".env")) {
    Write-ColorOutput Yellow "Warning: .env file not found. Creating from template..."
    
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-ColorOutput Yellow "Please edit .env file with your configuration"
    } else {
        # Create default .env file
        $envContent = @"
# AI Avatar Service Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514

# LiveKit Configuration
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://livekit:7880

# Performance Settings
TARGET_FPS=30
VIDEO_WIDTH=512
VIDEO_HEIGHT=512
TTS_SAMPLE_RATE=24000

# Logging
LOG_LEVEL=INFO
"@
        $envContent | Out-File -FilePath ".env" -Encoding utf8
        
        Write-ColorOutput Yellow "Created default .env file. Please edit with your API keys."
    }
}

Write-Output ""
Write-ColorOutput Blue "Environment: $Environment"
Write-ColorOutput Blue "Compose file: $ComposeFile"
Write-Output ""

# Build Docker images
Write-ColorOutput Green "Building Docker images..."
if ($UseDockerComposeV2) {
    docker compose -f $ComposeFile build
} else {
    docker-compose -f $ComposeFile build
}

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "Failed to build images"
    exit 1
}

# Create necessary directories
Write-ColorOutput Green "Creating directories..."
$directories = @(
    "assets\avatars",
    "assets\idle_loops",
    "assets\voice_samples",
    "logs",
    "models"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Production profile
$profiles = if ($Environment -eq "production") { "--profile production" } else { "" }

# Start services
Write-Output ""
Write-ColorOutput Green "Starting services..."

$composeCommand = if ($UseDockerComposeV2) { "docker compose" } else { "docker-compose" }
$startCommand = "$composeCommand -f $ComposeFile $profiles up $Detach"

Invoke-Expression $startCommand

if ($LASTEXITCODE -eq 0) {
    if ($Detach) {
        Write-Output ""
        Write-ColorOutput Green "Services started successfully!"
        Write-Output ""
        Write-Output "Useful commands:"
        Write-Output "  View logs:     docker-compose logs -f"
        Write-Output "  Stop services: docker-compose down"
        Write-Output "  Service status: docker-compose ps"
        Write-Output ""
        Write-Output "API endpoints:"
        Write-Output "  Health check:  http://localhost:8000/health"
        Write-Output "  API docs:      http://localhost:8000/docs"
        Write-Output "  WebSocket:     ws://localhost:8000/ws/"
    }
} else {
    Write-ColorOutput Red "Failed to start services"
    exit 1
}

