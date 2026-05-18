# Release Checklist

This document outlines the steps to create and publish a release.

## How the Release Workflow Works

The release workflow (`.github/workflows/release.yml`) is automatically registered when pushed to the repository. No additional GitHub UI configuration is needed.

**Triggers:**
- **Tag push** matching `v*` pattern (e.g., `v0.2.1`) - creates a production release
- **Manual dispatch** via GitHub Actions UI - creates a pre-release for testing

**What it does:**
- Builds the Windows executable using PyInstaller
- Creates a zip file with exe, LICENSE, README, and batch templates
- Uploads zip as build artifact (available in Actions for 30 days)
- Creates a GitHub release with the zip attached (when triggered by tag)

**Important:** The workflow handles zipping automatically. The `build.py` script creates the zip, and GitHub Actions uploads it directly (no double-zipping).

## Creating a Release

### Standard Release Process

1. **Update Version**:
   ```bash
   # Edit pyproject.toml and bump version number
   version = "X.Y.Z"
   ```

2. **Test Build Locally** (Optional):
   ```powershell
   uv run build.py
   # Test: .\dist\transcribe-windows-amd64.exe --help
   ```

3. **Commit and Tag**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump to vX.Y.Z"
   git push
   
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. **Workflow runs automatically**:
   - Monitor in Actions tab
   - Release appears in Releases page with zip attached
   - Build artifact also available in Actions run

### Manual Test Release (No Tag Required)

For testing the workflow without creating a release:

1. Go to **Actions** → **Build and Release**
2. Click **Run workflow**
3. **Leave "Create a GitHub release" unchecked**
4. Click **Run workflow**
5. Download artifact from the workflow run

To create a pre-release from manual dispatch:

1. Same as above but **check "Create a GitHub release"**
2. Creates a pre-release tagged as `manual-{run_number}`

## Post-Release Verification

- [ ] Check release appears on Releases page
- [ ] Download and test the zip file
- [ ] Verify exe runs: `transcribe.exe --help`

## Troubleshooting

### No Release Created After Tag Push

**Cause**: Manual workflow dispatch without checking "Create a GitHub release"

**Fix**: Either:
- Push a proper version tag (`vX.Y.Z`)
- OR manually dispatch with "Create a GitHub release" checked

### Double-Zipped Artifact

**Fixed in v0.2.1**: Workflow now uploads only the zip (not exe separately)

### Build Fails

- Verify `uv.lock` is committed
- Check Actions logs for specific errors
- Test locally first: `uv run build.py`
- Ensure all dependencies in `pyproject.toml`

### Permission Errors

If workflow can't create releases:
- Settings → Actions → General
- Enable "Read and write permissions" for workflows

