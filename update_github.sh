#!/bin/bash
# Script to update GitHub repository with latest changes

echo "==================================="
echo "GitHub Repository Update Script"
echo "==================================="

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository!"
    exit 1
fi

# Show current status
echo -e "\n📊 Current Status:"
git status --short

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then 
    echo "✅ No changes to commit. Repository is up to date!"
    exit 0
fi

# Show what files have changed
echo -e "\n📝 Modified files:"
git diff --name-only

# Add all changes
echo -e "\n➕ Adding all changes..."
git add -A

# Create commit message with timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
DEFAULT_MSG="Update COCONUT PPO implementation - $TIMESTAMP"

# Allow custom commit message
echo -e "\n💬 Enter commit message (or press Enter for default):"
echo "Default: $DEFAULT_MSG"
read -r CUSTOM_MSG

if [ -z "$CUSTOM_MSG" ]; then
    COMMIT_MSG="$DEFAULT_MSG"
else
    COMMIT_MSG="$CUSTOM_MSG"
fi

# Commit changes
echo -e "\n📦 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo -e "\n🚀 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo -e "\n✅ Successfully updated GitHub repository!"
    echo "View at: https://github.com/arccoxx/coconut"
else
    echo -e "\n❌ Push failed. You may need to set up authentication."
    echo "Try: git remote set-url origin https://YOUR_TOKEN@github.com/arccoxx/coconut.git"
fi

# Show latest commits
echo -e "\n📜 Latest commits:"
git log --oneline -5
