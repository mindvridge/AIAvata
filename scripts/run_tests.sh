#!/bin/bash
# Test runner script for AI Avatar Service
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI Avatar Service - Test Runner${NC}"
echo "=================================="

cd "$PROJECT_DIR"

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
MARKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --all)
            TEST_TYPE="all"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit         Run unit tests only"
            echo "  --e2e          Run E2E tests only"
            echo "  --integration  Run integration tests only"
            echo "  --all          Run all tests (default)"
            echo "  --coverage     Generate coverage report"
            echo "  -v, --verbose  Verbose output"
            echo "  -m, --markers  Run tests with specific markers"
            echo "  -h, --help     Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --unit                    # Run unit tests"
            echo "  $0 --e2e --coverage          # Run E2E tests with coverage"
            echo "  $0 -m 'not slow'             # Skip slow tests"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add verbosity
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

# Add markers
if [[ -n "$MARKERS" ]]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
fi

# Select test directory
case $TEST_TYPE in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/ --ignore=tests/e2e/"
        ;;
    e2e)
        echo -e "${BLUE}Running E2E tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/e2e/"
        ;;
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/test_integrations.py tests/test_pipeline.py"
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
esac

# Set environment variables for testing
export DEVICE=cpu
export ANTHROPIC_API_KEY=test_key
export LOG_LEVEL=WARNING

echo ""
echo -e "${YELLOW}Command: ${PYTEST_CMD}${NC}"
echo ""

# Run tests
eval $PYTEST_CMD
TEST_EXIT_CODE=$?

echo ""

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

# Show coverage report location if generated
if [[ "$COVERAGE" == true ]]; then
    echo ""
    echo -e "${BLUE}Coverage report: htmlcov/index.html${NC}"
fi

exit $TEST_EXIT_CODE
