#!/usr/bin/env bash
#
# Local code-health check using the CodeScene CLI.
#
# CodeScene is not part of the standard contributor workflow: the CLI requires a
# personal access token, so `make check` does not depend on it and CI does not
# run it. This script is for the maintainer, or for any contributor who happens
# to have a token.
#
# Usage:
#   scripts/codescene.sh                 # delta against the merge base with main
#   scripts/codescene.sh --base develop  # delta against another branch
#   scripts/codescene.sh --review        # also review each changed file in full
#   scripts/codescene.sh --all           # review every tracked Python file
#
# Reports are written to artifacts/code-quality/, which is git-ignored.

set -euo pipefail

readonly REQUIRED_MAJOR_MINOR="1.0"
readonly TESTED_VERSION="1.0.36"
readonly REPORT_DIR="artifacts/code-quality"

base_ref="main"
do_review=0
do_all=0

while [ $# -gt 0 ]; do
  case "$1" in
    --base) base_ref="${2:?--base needs a ref}"; shift 2 ;;
    --review) do_review=1; shift ;;
    --all) do_all=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- the CLI itself -------------------------------------------------------
if ! command -v cs >/dev/null 2>&1; then
  cat >&2 <<'EOF'
CodeScene CLI ("cs") not found on PATH.

Install it (a single binary) with:

  curl https://downloads.codescene.io/enterprise/cli/install-cs-tool.sh | sh

then re-run this script. Code health is a supplementary check: `make check` is
the gate every contributor is expected to pass.
EOF
  exit 127
fi

version="$(cs version 2>/dev/null | head -1 | sed -E 's/.*version ([0-9.]+).*/\1/')"
if [ -z "$version" ]; then
  echo "could not determine the CodeScene CLI version" >&2
  exit 1
fi
case "$version" in
  "$REQUIRED_MAJOR_MINOR".*) ;;
  *) echo "warning: cs $version found; this workflow was written against $TESTED_VERSION" >&2 ;;
esac

# --- authentication -------------------------------------------------------
# `cs` prints an instructional message and still exits 0 when unauthenticated,
# so a naive wrapper would report success having analysed nothing. Probe it
# first and treat that message as the failure it is. The token itself is never
# echoed, logged or written to a report.
if [ -z "${CS_ACCESS_TOKEN:-}" ]; then
  cat >&2 <<'EOF'
CS_ACCESS_TOKEN is not set, so the CodeScene CLI cannot analyse anything.

Create a personal access token:
  - codescene.io        https://codescene.io/users/me/pat
  - CodeScene Enterprise  <your-instance>/configuration/user/token

Then export it (and CS_ONPREM_URL for Enterprise) in your shell profile.
EOF
  exit 1
fi

probe="$(cs check "$0" 2>&1 || true)"
if printf '%s' "$probe" | grep -q "Personal Access Token"; then
  echo "CodeScene rejected the configured token; re-check CS_ACCESS_TOKEN." >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

# --- delta ----------------------------------------------------------------
# Compare against the merge base rather than the tip of the base branch, so the
# result describes this branch's own changes and not other people's.
compare_ref="$base_ref"
if merge_base="$(git merge-base HEAD "$base_ref" 2>/dev/null)"; then
  compare_ref="$merge_base"
fi

echo "==> cs delta against ${base_ref} (${compare_ref})"
cs delta "$compare_ref" | tee "$REPORT_DIR/delta.txt"

# --- optional per-file review --------------------------------------------
#
# Sources only. The demo's built bundle is committed and lives under
# serving/static/assets/, and CodeScene will happily analyse minified output:
# a single vite bundle scores 5.84 and contributes ten findings, which is both
# meaningless and enough to look like a real regression in a summary count.
is_source() {
  case "$1" in
    */static/assets/*|*/node_modules/*|demo/dist/*) return 1 ;;
    *.py|*.jsx|*.js) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "$do_all" -eq 1 ]; then
  # Tracked plus not-yet-committed sources, so a sweep run before the commit
  # lands still covers everything.
  candidates="$(
    git ls-files '*.py' '*.jsx' '*.js'
    git ls-files --others --exclude-standard '*.py' '*.jsx' '*.js'
  )"
elif [ "$do_review" -eq 1 ]; then
  candidates="$(
    git diff --name-only "$compare_ref" -- '*.py' '*.jsx' '*.js'
    git ls-files --others --exclude-standard '*.py' '*.jsx' '*.js'
  )"
else
  candidates=""
fi

files=""
for candidate in $candidates; do
  if [ -f "$candidate" ] && is_source "$candidate"; then
    files="$files $candidate"
  fi
done

if [ -n "$files" ]; then
  echo "==> cs check per file"
  : > "$REPORT_DIR/review.txt"
  for file in $files; do
    cs check "$file" >> "$REPORT_DIR/review.txt" 2>&1 || true
  done
  grep '^warn:' "$REPORT_DIR/review.txt" || echo "no code-health warnings"
fi

echo "==> reports in $REPORT_DIR/"
