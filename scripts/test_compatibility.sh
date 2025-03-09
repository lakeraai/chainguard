#!/usr/bin/env bash
set -euo pipefail

# Minimum supported versions from pyproject.toml
readonly LANGCHAIN_MIN="0.2.10"
readonly LANGCHAIN_CORE_MIN="0.2.43"
readonly LANGCHAIN_COMMUNITY_MIN="0.2.5"
readonly LANGCHAIN_OPENAI_MIN="0.1.25"

# Color definitions
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

print_heading() { echo -e "\n${YELLOW}$1${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}⨯ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

check_package_version() {
  local package="$1"
  local expected="$2"
  local version=$(poetry run pip show "$package" 2>/dev/null | grep "^Version" | cut -d ' ' -f 2)

  echo "$package: $version"

  if [[ "$version" == "$expected" ]]; then
    print_success "Version matches required: $version"
    return 0
  else
    print_error "Version mismatch: got $version, expected $expected"
    return 1
  fi
}

[[ -f "pyproject.toml" ]] || { echo -e "${RED}Error: Run from project root${NC}"; exit 1; }

echo -e "${YELLOW}======= Backwards Compatibility Test =======${NC}"
echo "Testing minimum versions: langchain ${LANGCHAIN_MIN}, core ${LANGCHAIN_CORE_MIN}, community ${LANGCHAIN_COMMUNITY_MIN}, openai ${LANGCHAIN_OPENAI_MIN}"

lakera_key="${LAKERA_GUARD_API_KEY:-}"
openai_key="${OPENAI_API_KEY:-}"

# If keys are not set in the environment, try loading from .env file as fallback
if [[ -z "$lakera_key" || -z "$openai_key" ]] && [[ -f ".env" ]]; then
    set -a && source .env && set +a
    lakera_key="${LAKERA_GUARD_API_KEY:-}"
    openai_key="${OPENAI_API_KEY:-}"
fi

[[ -n "$lakera_key" ]] || print_error "Missing LAKERA_GUARD_API_KEY"
[[ -n "$openai_key" ]] || print_error "Missing OPENAI_API_KEY"

# 1. Backup dependencies
print_heading "1. Backing up dependencies"
poetry run pip freeze > requirements.bak.txt

# 2. Uninstall existing packages
print_heading "2. Preparing environment"
for pkg in lakera-chainguard langchain langchain-core langchain-community langchain-openai langchain-text-splitters; do
  poetry run pip uninstall -y "$pkg" 2>/dev/null || true
done

# 3. Install minimum versions
print_heading "3. Installing minimum versions"
for pkg in "langchain-core==${LANGCHAIN_CORE_MIN}" \
           "langchain-community==${LANGCHAIN_COMMUNITY_MIN}" \
           "langchain-openai==${LANGCHAIN_OPENAI_MIN}" \
           "langchain==${LANGCHAIN_MIN}"; do
  poetry run pip install --no-deps "$pkg" > /dev/null
done
poetry run pip install --quiet -e .

# 4. Verify installed versions
print_heading "4. Verifying versions"
version_issues=false
for pkg_info in "langchain:$LANGCHAIN_MIN" \
                "langchain-core:$LANGCHAIN_CORE_MIN" \
                "langchain-community:$LANGCHAIN_COMMUNITY_MIN" \
                "langchain-openai:$LANGCHAIN_OPENAI_MIN"; do
  pkg=${pkg_info%%:*}
  version=${pkg_info#*:}
  check_package_version "$pkg" "$version" || version_issues=true
done

# 5. Run tests
print_heading "5. Running tests"
tests_failed=false
LAKERA_GUARD_API_KEY="$lakera_key" OPENAI_API_KEY="$openai_key" poetry run pytest -v || tests_failed=true

# 6. Restore original dependencies
print_heading "6. Restoring dependencies"
poetry run pip install -r requirements.bak.txt --quiet
rm requirements.bak.txt

# Report results
echo -e "\n${YELLOW}======= Results =======${NC}"
if [[ "$tests_failed" == true ]]; then
    print_error "Tests failed with minimum versions"
    exit 1
elif [[ "$version_issues" == true ]]; then
    print_warning "Tests passed but with version mismatches"
    exit 0
else
    print_success "All tests passed with minimum versions"
    exit 0
fi
