#!/usr/bin/env bash
set -euo pipefail

# Fast-forward main from upstream, then rebase the Jimsky Modal branch.
# Stops on conflicts. Does not auto-resolve or hard-reset custom work.

REPO_DIR="${1:-/opt/data/workspace/github-forks/AnyFlow}"
BRANCH="${JIMSKY_BRANCH:-jimsky/modal-skill}"

cd "$REPO_DIR"

git fetch upstream --prune
git fetch origin --prune

git checkout main
git merge --ff-only upstream/main
git push origin main

git checkout "$BRANCH"
git rebase main
git push --force-with-lease origin "$BRANCH"

echo "AnyFlow upstream sync complete for $BRANCH"
