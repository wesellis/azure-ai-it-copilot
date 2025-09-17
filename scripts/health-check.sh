#!/bin/bash

# Azure AI IT Copilot - Health Check Script
# This script performs comprehensive health checks on all system components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL=${API_URL:-"http://localhost:8000"}
DASHBOARD_URL=${DASHBOARD_URL:-"http://localhost:3000"}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}
CHECK_TIMEOUT=${CHECK_TIMEOUT:-30}

echo -e "${BLUE}Azure AI IT Copilot - System Health Check${NC}"
echo "=================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting health check..."
echo ""

# Health check results
declare -A RESULTS
OVERALL_STATUS="HEALTHY"

# Function to log results
log_result() {
    local component=$1
    local status=$2
    local message=$3

    RESULTS[$component]=$status

    if [ "$status" = "PASS" ]; then
        echo -e "  ✅ ${GREEN}$component${NC}: $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "  ⚠️  ${YELLOW}$component${NC}: $message"
    else
        echo -e "  ❌ ${RED}$component${NC}: $message"
        OVERALL_STATUS="UNHEALTHY"
    fi
}

# Function to check if a service is responding
check_http_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    if command -v curl &> /dev/null; then
        response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $CHECK_TIMEOUT "$url" 2>/dev/null)
        if [ "$response" = "$expected_status" ]; then
            log_result "$name" "PASS" "Service responding (HTTP $response)"
        else
            log_result "$name" "FAIL" "Service not responding (HTTP $response)"
        fi
    else
        log_result "$name" "WARN" "curl not available, cannot check HTTP service"
    fi
}

# Function to check if a port is open
check_port() {
    local name=$1
    local host=$2
    local port=$3

    if command -v nc &> /dev/null; then
        if nc -z -w5 "$host" "$port" 2>/dev/null; then
            log_result "$name" "PASS" "Port $port is open on $host"
        else
            log_result "$name" "FAIL" "Port $port is not reachable on $host"
        fi
    elif command -v telnet &> /dev/null; then
        if timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            log_result "$name" "PASS" "Port $port is open on $host"
        else
            log_result "$name" "FAIL" "Port $port is not reachable on $host"
        fi
    else
        log_result "$name" "WARN" "Neither nc nor telnet available, cannot check port"
    fi
}

# Function to check disk space
check_disk_space() {
    local threshold=${1:-90}

    while read -r filesystem usage mountpoint; do
        if [ "$mountpoint" = "/" ] || [ "$mountpoint" = "/tmp" ] || [[ "$mountpoint" == */data ]]; then
            usage_num=${usage%\%}
            if [ "$usage_num" -gt "$threshold" ]; then
                log_result "Disk Space ($mountpoint)" "FAIL" "Usage at $usage (threshold: ${threshold}%)"
            elif [ "$usage_num" -gt 80 ]; then
                log_result "Disk Space ($mountpoint)" "WARN" "Usage at $usage (threshold: ${threshold}%)"
            else
                log_result "Disk Space ($mountpoint)" "PASS" "Usage at $usage"
            fi
        fi
    done < <(df -h | awk 'NR>1 {print $1, $5, $6}')
}

# Function to check memory usage
check_memory() {
    if command -v free &> /dev/null; then
        local mem_usage=$(free | awk 'FNR==2{printf "%.0f", $3/($3+$4)*100}')
        if [ "$mem_usage" -gt 90 ]; then
            log_result "Memory Usage" "FAIL" "Usage at ${mem_usage}%"
        elif [ "$mem_usage" -gt 80 ]; then
            log_result "Memory Usage" "WARN" "Usage at ${mem_usage}%"
        else
            log_result "Memory Usage" "PASS" "Usage at ${mem_usage}%"
        fi
    else
        log_result "Memory Usage" "WARN" "free command not available"
    fi
}

# Function to check CPU load
check_cpu_load() {
    if command -v uptime &> /dev/null; then
        local load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
        local cpu_count=$(nproc 2>/dev/null || echo "1")
        local load_percent=$(echo "scale=0; $load * 100 / $cpu_count" | bc -l 2>/dev/null || echo "0")

        if [ "$load_percent" -gt 90 ]; then
            log_result "CPU Load" "FAIL" "Load average: $load (${load_percent}% of capacity)"
        elif [ "$load_percent" -gt 70 ]; then
            log_result "CPU Load" "WARN" "Load average: $load (${load_percent}% of capacity)"
        else
            log_result "CPU Load" "PASS" "Load average: $load (${load_percent}% of capacity)"
        fi
    else
        log_result "CPU Load" "WARN" "uptime command not available"
    fi
}

# Function to check Python dependencies
check_python_deps() {
    if command -v python3 &> /dev/null; then
        local missing_deps=()

        # Check critical Python packages
        for package in "fastapi" "uvicorn" "redis" "azure-identity" "azure-mgmt-resource" "langchain-openai"; do
            if ! python3 -c "import ${package//-/_}" 2>/dev/null; then
                missing_deps+=("$package")
            fi
        done

        if [ ${#missing_deps[@]} -eq 0 ]; then
            log_result "Python Dependencies" "PASS" "All critical packages available"
        else
            log_result "Python Dependencies" "FAIL" "Missing packages: ${missing_deps[*]}"
        fi
    else
        log_result "Python Dependencies" "FAIL" "Python 3 not available"
    fi
}

# Function to check Node.js dependencies
check_node_deps() {
    if command -v npm &> /dev/null; then
        if [ -f "dashboard/package.json" ]; then
            cd dashboard
            if npm list --depth=0 &>/dev/null; then
                log_result "Node.js Dependencies" "PASS" "All packages installed"
            else
                log_result "Node.js Dependencies" "WARN" "Some packages may be missing (run npm install)"
            fi
            cd ..
        else
            log_result "Node.js Dependencies" "WARN" "package.json not found in dashboard/"
        fi
    else
        log_result "Node.js Dependencies" "WARN" "npm not available"
    fi
}

# Function to check environment variables
check_environment() {
    local missing_vars=()
    local important_vars=("AZURE_SUBSCRIPTION_ID" "AZURE_TENANT_ID" "AZURE_CLIENT_ID" "AZURE_OPENAI_ENDPOINT")

    for var in "${important_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -eq 0 ]; then
        log_result "Environment Variables" "PASS" "All critical variables set"
    else
        log_result "Environment Variables" "WARN" "Missing variables: ${missing_vars[*]}"
    fi
}

# Function to check log files
check_logs() {
    local log_dirs=("logs" "automation_engine/powershell/logs" "/var/log")
    local recent_errors=0

    for log_dir in "${log_dirs[@]}"; do
        if [ -d "$log_dir" ]; then
            # Check for recent errors in log files
            recent_errors=$(find "$log_dir" -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL\|FATAL" {} \; 2>/dev/null | wc -l)
        fi
    done

    if [ "$recent_errors" -eq 0 ]; then
        log_result "Log Analysis" "PASS" "No recent errors found"
    elif [ "$recent_errors" -lt 5 ]; then
        log_result "Log Analysis" "WARN" "$recent_errors files with recent errors"
    else
        log_result "Log Analysis" "FAIL" "$recent_errors files with recent errors"
    fi
}

# Function to check API endpoints
check_api_endpoints() {
    local endpoints=(
        "/health:200"
        "/api/v1/auth/validate:401"  # Should return 401 without auth
        "/metrics:200"
    )

    for endpoint_info in "${endpoints[@]}"; do
        local endpoint=${endpoint_info%:*}
        local expected=${endpoint_info#*:}
        check_http_service "API Endpoint $endpoint" "$API_URL$endpoint" "$expected"
    done
}

# Function to check database/cache connectivity
check_external_services() {
    # Check Redis
    check_port "Redis Connection" "$REDIS_HOST" "$REDIS_PORT"

    # Try to ping Redis if redis-cli is available
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &>/dev/null; then
            log_result "Redis Ping" "PASS" "Redis responding to ping"
        else
            log_result "Redis Ping" "FAIL" "Redis not responding to ping"
        fi
    fi
}

# Function to check Azure connectivity
check_azure_connectivity() {
    if command -v az &> /dev/null; then
        if az account show &>/dev/null; then
            log_result "Azure CLI" "PASS" "Authenticated to Azure"
        else
            log_result "Azure CLI" "WARN" "Not authenticated to Azure (az login required)"
        fi
    else
        log_result "Azure CLI" "WARN" "Azure CLI not installed"
    fi
}

# Function to check file permissions
check_file_permissions() {
    local critical_files=(
        "scripts/deploy.sh:755"
        "scripts/dev.sh:755"
        "automation_engine/powershell:755"
    )

    local perm_issues=()

    for file_info in "${critical_files[@]}"; do
        local file=${file_info%:*}
        local expected_perm=${file_info#*:}

        if [ -e "$file" ]; then
            local actual_perm=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%A" "$file" 2>/dev/null || echo "unknown")
            if [ "$actual_perm" != "$expected_perm" ] && [ "$actual_perm" != "unknown" ]; then
                perm_issues+=("$file:$actual_perm")
            fi
        fi
    done

    if [ ${#perm_issues[@]} -eq 0 ]; then
        log_result "File Permissions" "PASS" "All permissions correct"
    else
        log_result "File Permissions" "WARN" "Permission issues: ${perm_issues[*]}"
    fi
}

# Function to check Docker (if used)
check_docker() {
    if command -v docker &> /dev/null; then
        if docker ps &>/dev/null; then
            local running_containers=$(docker ps --format "table {{.Names}}" | grep -v NAMES | wc -l)
            log_result "Docker" "PASS" "$running_containers containers running"
        else
            log_result "Docker" "WARN" "Docker daemon not accessible"
        fi
    else
        log_result "Docker" "WARN" "Docker not installed"
    fi
}

# Main health check execution
echo -e "${YELLOW}System Resources:${NC}"
check_disk_space 85
check_memory
check_cpu_load
echo ""

echo -e "${YELLOW}Dependencies:${NC}"
check_python_deps
check_node_deps
check_environment
echo ""

echo -e "${YELLOW}Services:${NC}"
check_api_endpoints
check_external_services
echo ""

echo -e "${YELLOW}Infrastructure:${NC}"
check_azure_connectivity
check_docker
echo ""

echo -e "${YELLOW}Configuration:${NC}"
check_file_permissions
check_logs
echo ""

# Summary
echo "=================================================="
echo -e "${BLUE}Health Check Summary:${NC}"
echo ""

# Count results
total_checks=0
passed_checks=0
warned_checks=0
failed_checks=0

for component in "${!RESULTS[@]}"; do
    ((total_checks++))
    case ${RESULTS[$component]} in
        "PASS") ((passed_checks++)) ;;
        "WARN") ((warned_checks++)) ;;
        "FAIL") ((failed_checks++)) ;;
    esac
done

echo "Total Checks: $total_checks"
echo -e "Passed: ${GREEN}$passed_checks${NC}"
echo -e "Warnings: ${YELLOW}$warned_checks${NC}"
echo -e "Failed: ${RED}$failed_checks${NC}"
echo ""

# Overall status
if [ "$OVERALL_STATUS" = "HEALTHY" ]; then
    if [ "$warned_checks" -gt 0 ]; then
        echo -e "Overall Status: ${YELLOW}HEALTHY WITH WARNINGS${NC}"
        exit_code=1
    else
        echo -e "Overall Status: ${GREEN}HEALTHY${NC}"
        exit_code=0
    fi
else
    echo -e "Overall Status: ${RED}UNHEALTHY${NC}"
    exit_code=2
fi

echo ""
echo "Health check completed at $(date '+%Y-%m-%d %H:%M:%S')"

# Generate JSON report if requested
if [ "$1" = "--json" ]; then
    cat > health-check-report.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "overall_status": "$OVERALL_STATUS",
    "summary": {
        "total_checks": $total_checks,
        "passed": $passed_checks,
        "warnings": $warned_checks,
        "failed": $failed_checks
    },
    "results": {
EOF

    first=true
    for component in "${!RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> health-check-report.json
        fi
        echo "        \"$component\": \"${RESULTS[$component]}\"" >> health-check-report.json
    done

    cat >> health-check-report.json << EOF
    }
}
EOF
    echo "JSON report saved to health-check-report.json"
fi

exit $exit_code