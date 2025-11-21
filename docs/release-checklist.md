# Release Checklist

This document outlines the manual steps required to prepare and publish a release on GitHub.

## Pre-Release Steps

### 0. Clean Up Legacy Files (One-Time Setup)

Before the first release, move legacy documentation to a separate branch:

```bash
# Create and switch to legacy branch
git checkout -b legacy-docs

# Move legacy files (if not already done)
git mv legacy/ legacy-docs/
git mv doc_build_fix_walkthrough.md legacy-docs/ 2>/dev/null || true
git mv walkthrough.md legacy-docs/ 2>/dev/null || true
git mv REFACTORING_PLAN.md legacy-docs/ 2>/dev/null || true
git mv features.md legacy-docs/ 2>/dev/null || true
git mv tui_implementation_plan.md legacy-docs/ 2>/dev/null || true

# Commit and push
git commit -m "Archive legacy documentation"
git push origin legacy-docs

# Switch back to main
git checkout main

# Remove legacy files from main (they're now in legacy-docs branch)
# Note: Only do this if you're sure everything is in legacy-docs branch
```

**Note**: This is a one-time operation. After this, the `legacy-docs` branch preserves the history while `main` stays clean.

### 1. Repository Settings

- [ ] **Enable GitHub Actions**: Go to Settings → Actions → General
  - Allow all actions and reusable workflows
  - Enable "Read and write permissions" for workflows

- [ ] **Branch Protection** (Optional but recommended):
  - Go to Settings → Branches
  - Add rule for `main` branch
  - Enable "Require pull request reviews before merging"
  - Enable "Require status checks to pass before merging"
  - Select the `build` job from the CI workflow

### 2. Secrets Configuration

If you need to use any secrets in workflows (e.g., for publishing to package registries):
- [ ] Go to Settings → Secrets and variables → Actions
- [ ] Add any required secrets (e.g., `PYPI_API_TOKEN`, `NPM_TOKEN`)

**Note**: For basic releases, no secrets are needed as GitHub automatically provides `GITHUB_TOKEN`.

### 3. Release Preparation

- [ ] **Update Version Numbers**:
  - Update version in `pyproject.toml` (if applicable)
  - Update `CHANGELOG.md` with release notes
  - Update `README.md` if there are breaking changes

- [ ] **Test the Build Locally**:
  ```powershell
  uv run build.py
  ```
  - Verify the executable is created in `dist/`
  - Verify the zip archive is created
  - Test the executable: `.\dist\transcribe-windows-amd64.exe --help`

- [ ] **Commit All Changes**:
  ```bash
  git add .
  git commit -m "Prepare for release vX.Y.Z"
  git push origin main
  ```

## Creating a Release

### Option 1: Using Git Tags (Recommended)

1. **Create and Push Tag**:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

2. **GitHub Actions will automatically**:
   - Build the executable
   - Create a zip archive
   - Create a GitHub Release with the artifacts

### Option 2: Using GitHub UI

1. Go to **Releases** → **Draft a new release**
2. Choose a tag (create new if needed): `v1.0.0`
3. Fill in release title and description (copy from CHANGELOG.md)
4. Click **Publish release**
5. GitHub Actions will build and attach artifacts

### Option 3: Manual Workflow Dispatch

1. Go to **Actions** → **Build and Release** workflow
2. Click **Run workflow**
3. Check "Create a GitHub release"
4. Click **Run workflow**
5. Wait for the build to complete
6. The release will be created automatically

## Post-Release Steps

- [ ] **Verify Release**:
  - Check that the release appears on the Releases page
  - Verify the zip file is attached
  - Download and test the release artifact

- [ ] **Update Documentation**:
  - Update any installation instructions if needed
  - Update any version-specific documentation

- [ ] **Announce Release** (Optional):
  - Post on social media
  - Update project website
  - Notify users via issues/discussions

## Troubleshooting

### Build Fails in GitHub Actions

- Check the Actions tab for error messages
- Verify `uv.lock` is committed
- Ensure all dependencies are properly specified in `pyproject.toml`

### Release Not Created

- Verify the workflow has permission to create releases
- Check that the tag format matches `v*` (e.g., `v1.0.0`)
- Review workflow logs for errors

### Executable Not Working

- Test locally first before releasing
- Check that all dependencies are included in PyInstaller spec
- Verify hidden imports are specified in `build.py`

## Future Enhancements

Consider automating:
- [ ] Automatic version bumping
- [ ] Automatic CHANGELOG generation
- [ ] Publishing to package registries (PyPI, etc.)
- [ ] Multi-platform builds (Linux, macOS)
- [ ] Code signing for Windows executables

