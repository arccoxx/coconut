"""Script to update GitHub with changes"""

import subprocess
import os

def run_command(cmd):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def update_github():
    # Check status
    print("=== Git Status ===")
    run_command("git status")
    
    # Add files
    print("\n=== Adding files ===")
    run_command("git add -A")
    
    # Commit
    print("\n=== Committing ===")
    commit_msg = "Update COCONUT PPO: Fix device placement, add Llama3 support, integrate continuous reasoning navigator"
    run_command(f'git commit -m "{commit_msg}"')
    
    # Push
    print("\n=== Pushing to GitHub ===")
    if run_command("git push origin main"):
        print("✅ Successfully pushed to GitHub!")
    else:
        print("❌ Push failed. You may need to set up authentication.")
        print("Try running:")
        print("git remote set-url origin https://YOUR_TOKEN@github.com/arccoxx/coconut.git")

if __name__ == "__main__":
    update_github()
