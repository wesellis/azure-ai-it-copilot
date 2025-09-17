#!/bin/bash

# Azure AI IT Copilot - Test Execution Script
# Comprehensive test runner with coverage reporting and performance analysis

set -e  # Exit on any error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${PROJECT_ROOT}/tests"
COVERAGE_DIR="${PROJECT_ROOT}/htmlcov"
REPORTS_DIR="${PROJECT_ROOT}/test-reports"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark-results"

# Default configuration
COVERAGE_THRESHOLD=85
PARALLEL_WORKERS="auto"
TEST_TIMEOUT=300
LOAD_TEST_DURATION=60
LOAD_TEST_USERS=20

# Parse command line arguments
VERBOSE=false
COVERAGE_ONLY=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
LOAD_ONLY=false
PERFORMANCE_ONLY=false
SKIP_LOAD=false
SKIP_PERFORMANCE=false
CLEAN_CACHE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -c, --coverage-only     Run only coverage analysis"
    echo "  -u, --unit-only         Run only unit tests"
    echo "  -i, --integration-only  Run only integration tests"
    echo "  -l, --load-only         Run only load tests"
    echo "  -p, --performance-only  Run only performance tests"
    echo "  --skip-load             Skip load tests"
    echo "  --skip-performance      Skip performance tests"
    echo "  --clean-cache           Clean pytest cache before running"
    echo "  --coverage-threshold N  Set coverage threshold (default: 85)"
    echo "  --parallel-workers N    Set number of parallel workers (default: auto)"
    echo "  --test-timeout N        Set test timeout in seconds (default: 300)"
    echo "  --load-duration N       Set load test duration in seconds (default: 60)"
    echo "  --load-users N          Set number of load test users (default: 20)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage-only)
            COVERAGE_ONLY=true
            shift
            ;;
        -u|--unit-only)
            UNIT_ONLY=true
            shift
            ;;
        -i|--integration-only)
            INTEGRATION_ONLY=true
            shift
            ;;
        -l|--load-only)
            LOAD_ONLY=true
            shift
            ;;
        -p|--performance-only)
            PERFORMANCE_ONLY=true
            shift
            ;;
        --skip-load)
            SKIP_LOAD=true
            shift
            ;;
        --skip-performance)
            SKIP_PERFORMANCE=true
            shift
            ;;
        --clean-cache)
            CLEAN_CACHE=true
            shift
            ;;
        --coverage-threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        --parallel-workers)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        --test-timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --load-duration)
            LOAD_TEST_DURATION="$2"
            shift 2
            ;;
        --load-users)
            LOAD_TEST_USERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Utility functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_warning "Not running in a virtual environment"
    else
        print_success "Virtual environment detected: ${VIRTUAL_ENV}"
    fi

    # Check required tools
    local required_tools=("pytest" "coverage" "locust")
    for tool in "${required_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            print_success "$tool is available"
        else
            print_error "$tool is not installed"
            echo "Please install test dependencies: pip install -r tests/test_requirements.txt"
            exit 1
        fi
    done
}

setup_directories() {
    print_header "Setting Up Directories"

    # Create necessary directories
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$BENCHMARK_DIR"
    mkdir -p "$COVERAGE_DIR"

    print_success "Test directories created"
}

clean_cache() {
    if [[ "$CLEAN_CACHE" == true ]]; then
        print_header "Cleaning Cache"

        # Remove pytest cache
        if [[ -d "${PROJECT_ROOT}/.pytest_cache" ]]; then
            rm -rf "${PROJECT_ROOT}/.pytest_cache"
            print_success "Pytest cache cleared"
        fi

        # Remove coverage data
        if [[ -f "${PROJECT_ROOT}/.coverage" ]]; then
            rm -f "${PROJECT_ROOT}/.coverage"
            print_success "Coverage data cleared"
        fi

        # Remove __pycache__ directories
        find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        print_success "Python cache cleared"
    fi
}

run_unit_tests() {
    print_header "Running Unit Tests"

    local pytest_args=(
        "${TEST_DIR}/unit/"
        "-v"
        "--tb=short"
        "--strict-markers"
        "--color=yes"
        "-n" "$PARALLEL_WORKERS"
        "--timeout=$TEST_TIMEOUT"
        "--junitxml=${REPORTS_DIR}/unit-test-results.xml"
        "--html=${REPORTS_DIR}/unit-test-report.html"
        "--self-contained-html"
    )

    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("-vv" "--tb=long")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "Unit tests passed"
        return 0
    else
        print_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    print_header "Running Integration Tests"

    local pytest_args=(
        "${TEST_DIR}/integration/"
        "-v"
        "--tb=short"
        "--strict-markers"
        "--color=yes"
        "-n" "$PARALLEL_WORKERS"
        "--timeout=$TEST_TIMEOUT"
        "-m" "integration"
        "--junitxml=${REPORTS_DIR}/integration-test-results.xml"
        "--html=${REPORTS_DIR}/integration-test-report.html"
        "--self-contained-html"
    )

    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("-vv" "--tb=long")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "Integration tests passed"
        return 0
    else
        print_error "Integration tests failed"
        return 1
    fi
}

run_api_tests() {
    print_header "Running API Tests"

    local pytest_args=(
        "${TEST_DIR}/api/"
        "-v"
        "--tb=short"
        "--strict-markers"
        "--color=yes"
        "-n" "$PARALLEL_WORKERS"
        "--timeout=$TEST_TIMEOUT"
        "-m" "api"
        "--junitxml=${REPORTS_DIR}/api-test-results.xml"
        "--html=${REPORTS_DIR}/api-test-report.html"
        "--self-contained-html"
    )

    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("-vv" "--tb=long")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "API tests passed"
        return 0
    else
        print_error "API tests failed"
        return 1
    fi
}

run_database_tests() {
    print_header "Running Database Tests"

    local pytest_args=(
        "${TEST_DIR}/database/"
        "-v"
        "--tb=short"
        "--strict-markers"
        "--color=yes"
        "-n" "$PARALLEL_WORKERS"
        "--timeout=$TEST_TIMEOUT"
        "-m" "database"
        "--junitxml=${REPORTS_DIR}/database-test-results.xml"
        "--html=${REPORTS_DIR}/database-test-report.html"
        "--self-contained-html"
    )

    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("-vv" "--tb=long")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "Database tests passed"
        return 0
    else
        print_error "Database tests failed"
        return 1
    fi
}

run_coverage_analysis() {
    print_header "Running Coverage Analysis"

    local pytest_args=(
        "$TEST_DIR"
        "--cov=."
        "--cov-report=html:${COVERAGE_DIR}"
        "--cov-report=xml:${PROJECT_ROOT}/coverage.xml"
        "--cov-report=json:${PROJECT_ROOT}/coverage.json"
        "--cov-report=term-missing"
        "--cov-branch"
        "--cov-fail-under=$COVERAGE_THRESHOLD"
        "-n" "$PARALLEL_WORKERS"
        "--tb=short"
        "-q"
    )

    # Exclude load and performance tests from coverage if skipped
    if [[ "$SKIP_LOAD" == true ]]; then
        pytest_args+=("--ignore=${TEST_DIR}/load/")
    fi

    if [[ "$SKIP_PERFORMANCE" == true ]]; then
        pytest_args+=("--ignore=${TEST_DIR}/performance/")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "Coverage analysis completed"
        print_success "Coverage report available at: ${COVERAGE_DIR}/index.html"
        return 0
    else
        print_error "Coverage analysis failed - coverage below threshold"
        return 1
    fi
}

run_load_tests() {
    print_header "Running Load Tests"

    # Check if application is running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_warning "Application not running on localhost:8000"
        print_warning "Please start the application before running load tests"
        return 1
    fi

    local locust_args=(
        "-f" "${TEST_DIR}/load/test_ai_orchestrator_load.py"
        "--host=http://localhost:8000"
        "--users=$LOAD_TEST_USERS"
        "--spawn-rate=5"
        "--run-time=${LOAD_TEST_DURATION}s"
        "--headless"
        "--html=${REPORTS_DIR}/load-test-report.html"
        "--csv=${REPORTS_DIR}/load-test"
    )

    if locust "${locust_args[@]}"; then
        print_success "Load tests completed"
        print_success "Load test report available at: ${REPORTS_DIR}/load-test-report.html"
        return 0
    else
        print_error "Load tests failed"
        return 1
    fi
}

run_performance_tests() {
    print_header "Running Performance Tests"

    local pytest_args=(
        "${TEST_DIR}/performance/"
        "-v"
        "--tb=short"
        "--strict-markers"
        "--color=yes"
        "--timeout=$TEST_TIMEOUT"
        "-m" "performance"
        "--benchmark-sort=mean"
        "--benchmark-save=${BENCHMARK_DIR}/benchmark_results"
        "--benchmark-save-data"
        "--benchmark-histogram=${BENCHMARK_DIR}/benchmark_histogram"
        "--junitxml=${REPORTS_DIR}/performance-test-results.xml"
        "--html=${REPORTS_DIR}/performance-test-report.html"
        "--self-contained-html"
    )

    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("-vv" "--tb=long")
    fi

    if pytest "${pytest_args[@]}"; then
        print_success "Performance tests completed"
        print_success "Benchmark results saved to: ${BENCHMARK_DIR}/"
        return 0
    else
        print_error "Performance tests failed"
        return 1
    fi
}

generate_summary_report() {
    print_header "Test Summary Report"

    local report_file="${REPORTS_DIR}/test-summary.txt"

    {
        echo "Azure AI IT Copilot - Test Execution Summary"
        echo "============================================="
        echo "Execution Date: $(date)"
        echo "Project Root: $PROJECT_ROOT"
        echo ""

        echo "Configuration:"
        echo "  Coverage Threshold: ${COVERAGE_THRESHOLD}%"
        echo "  Parallel Workers: $PARALLEL_WORKERS"
        echo "  Test Timeout: ${TEST_TIMEOUT}s"
        echo ""

        if [[ "$SKIP_LOAD" != true && "$LOAD_ONLY" != true ]]; then
            echo "Load Test Configuration:"
            echo "  Duration: ${LOAD_TEST_DURATION}s"
            echo "  Users: $LOAD_TEST_USERS"
            echo ""
        fi

        echo "Test Results:"
        echo "============="

        # Count test results from XML files
        if [[ -f "${REPORTS_DIR}/unit-test-results.xml" ]]; then
            local unit_tests=$(grep -o 'tests="[0-9]*"' "${REPORTS_DIR}/unit-test-results.xml" | grep -o '[0-9]*' || echo "0")
            local unit_failures=$(grep -o 'failures="[0-9]*"' "${REPORTS_DIR}/unit-test-results.xml" | grep -o '[0-9]*' || echo "0")
            echo "  Unit Tests: $unit_tests tests, $unit_failures failures"
        fi

        if [[ -f "${REPORTS_DIR}/integration-test-results.xml" ]]; then
            local integration_tests=$(grep -o 'tests="[0-9]*"' "${REPORTS_DIR}/integration-test-results.xml" | grep -o '[0-9]*' || echo "0")
            local integration_failures=$(grep -o 'failures="[0-9]*"' "${REPORTS_DIR}/integration-test-results.xml" | grep -o '[0-9]*' || echo "0")
            echo "  Integration Tests: $integration_tests tests, $integration_failures failures"
        fi

        if [[ -f "${REPORTS_DIR}/api-test-results.xml" ]]; then
            local api_tests=$(grep -o 'tests="[0-9]*"' "${REPORTS_DIR}/api-test-results.xml" | grep -o '[0-9]*' || echo "0")
            local api_failures=$(grep -o 'failures="[0-9]*"' "${REPORTS_DIR}/api-test-results.xml" | grep -o '[0-9]*' || echo "0")
            echo "  API Tests: $api_tests tests, $api_failures failures"
        fi

        if [[ -f "${REPORTS_DIR}/database-test-results.xml" ]]; then
            local db_tests=$(grep -o 'tests="[0-9]*"' "${REPORTS_DIR}/database-test-results.xml" | grep -o '[0-9]*' || echo "0")
            local db_failures=$(grep -o 'failures="[0-9]*"' "${REPORTS_DIR}/database-test-results.xml" | grep -o '[0-9]*' || echo "0")
            echo "  Database Tests: $db_tests tests, $db_failures failures"
        fi

        if [[ -f "${REPORTS_DIR}/performance-test-results.xml" ]]; then
            local perf_tests=$(grep -o 'tests="[0-9]*"' "${REPORTS_DIR}/performance-test-results.xml" | grep -o '[0-9]*' || echo "0")
            local perf_failures=$(grep -o 'failures="[0-9]*"' "${REPORTS_DIR}/performance-test-results.xml" | grep -o '[0-9]*' || echo "0")
            echo "  Performance Tests: $perf_tests tests, $perf_failures failures"
        fi

        echo ""
        echo "Reports Generated:"
        echo "=================="
        echo "  Coverage Report: ${COVERAGE_DIR}/index.html"
        echo "  Test Reports: ${REPORTS_DIR}/"

        if [[ -f "${REPORTS_DIR}/load-test-report.html" ]]; then
            echo "  Load Test Report: ${REPORTS_DIR}/load-test-report.html"
        fi

        if [[ -d "$BENCHMARK_DIR" ]]; then
            echo "  Benchmark Results: ${BENCHMARK_DIR}/"
        fi

    } > "$report_file"

    cat "$report_file"
    print_success "Summary report saved to: $report_file"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"

    local exit_code=0
    local start_time=$(date +%s)

    check_dependencies
    setup_directories
    clean_cache

    # Execute tests based on options
    if [[ "$COVERAGE_ONLY" == true ]]; then
        run_coverage_analysis || exit_code=1
    elif [[ "$UNIT_ONLY" == true ]]; then
        run_unit_tests || exit_code=1
    elif [[ "$INTEGRATION_ONLY" == true ]]; then
        run_integration_tests || exit_code=1
    elif [[ "$LOAD_ONLY" == true ]]; then
        run_load_tests || exit_code=1
    elif [[ "$PERFORMANCE_ONLY" == true ]]; then
        run_performance_tests || exit_code=1
    else
        # Run all tests
        run_unit_tests || exit_code=1
        run_integration_tests || exit_code=1
        run_api_tests || exit_code=1
        run_database_tests || exit_code=1

        if [[ "$SKIP_LOAD" != true ]]; then
            run_load_tests || exit_code=1
        fi

        if [[ "$SKIP_PERFORMANCE" != true ]]; then
            run_performance_tests || exit_code=1
        fi

        run_coverage_analysis || exit_code=1
    fi

    # Generate summary report
    generate_summary_report

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    print_header "Execution Complete"
    echo "Total execution time: ${duration} seconds"

    if [[ $exit_code -eq 0 ]]; then
        print_success "All tests passed successfully!"
    else
        print_error "Some tests failed. Check the reports for details."
    fi

    exit $exit_code
}

# Execute main function
main "$@"