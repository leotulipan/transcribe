# Release SOP — Audio Transcribe

How to cut a versioned release that builds and publishes Windows + macOS
binaries automatically via GitHub Actions.

- **Trigger:** pushing a `v*` git tag (e.g. `v0.11.0`).
- **Build matrix:** `windows-latest` (amd64) and `macos-14` (Apple Silicon /
  arm64). Fyne links CGO, so each OS builds on its own native runner. Intel
  macOS is intentionally not built: GitHub's hosted `macos-13` runners queued
  indefinitely in practice. To add Intel back later, cross-compile on the
  `macos-14` runner with `CGO_ENABLED=1 GOARCH=amd64 CC="clang -arch x86_64"`.
- **Output:** one GitHub Release with the Windows installer, a portable
  Windows zip, two macOS tarballs, `.sha256` sidecars for each, and
  auto-generated release notes.
- **Signing:** none. Artifacts are unsigned; see
  [`end-user-install-notes.md`](end-user-install-notes.md).

The workflows live at:

- `.github/workflows/ci.yml` — build + `go vet` + `go test` on Windows and
  macOS for every push/PR.
- `.github/workflows/release.yml` — the release pipeline described here.

---

## Part A — One-time GitHub web-UI setup

Do this once per repository. Everything is under **Settings** on
<https://github.com/leotulipan/transcribe>.

1. **Enable Actions** (public repos: on by default).
   - Settings → **Actions** → **General** → *Actions permissions*:
     **Allow all actions and reusable workflows**.
   - This repo uses these marketplace actions: `actions/checkout`,
     `actions/setup-go`, `actions/upload-artifact`,
     `actions/download-artifact`, `softprops/action-gh-release`. If you ever
     switch to the restricted "Allow select actions" mode, allow at least
     `actions/*` and `softprops/action-gh-release@*`.

2. **Let the workflow create releases.**
   - Settings → **Actions** → **General** → *Workflow permissions*:
     select **Read and write permissions**, then **Save**.
   - The release workflow also declares `permissions: contents: write`
     itself, but setting the repo default to read/write avoids any 403 on
     the first run. Leave "Allow GitHub Actions to create and approve pull
     requests" **off** — not needed.

3. **Secrets.** None required for unsigned builds. (When you later add code
   signing, this is where the certs/passwords go — Settings → *Secrets and
   variables* → **Actions**.)

That's the entire UI configuration. No runners to register (GitHub-hosted),
no environments, no protected tags needed.

---

## Part B — Cutting a release (the recurring steps)

Run these from a clean `main` that is pushed and green in CI.

1. **Pick the version.** Semantic Versioning, `vMAJOR.MINOR.PATCH`. New
   features → bump MINOR; fixes only → bump PATCH.

2. **Update `CHANGELOG.md`.**
   - Rename the `## [Unreleased]` heading's contents to
     `## [X.Y.Z] - YYYY-MM-DD` and leave a fresh empty `## [Unreleased]`
     above it.
   - At the bottom, add/refresh the compare links:
     ```
     [Unreleased]: https://github.com/leotulipan/transcribe/compare/vX.Y.Z...HEAD
     [X.Y.Z]: https://github.com/leotulipan/transcribe/compare/vPREV...vX.Y.Z
     ```

3. **Commit and push `main`.**
   ```bash
   git add CHANGELOG.md
   git commit -m "chore(release): X.Y.Z"
   git push origin main
   ```

4. **Tag and push the tag.** The tag must point at the commit that contains
   the corrected workflows (otherwise the *old* workflow at that commit
   runs).
   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Watch the run.** Actions tab → **Release**. The `windows` and
   `macos (arm64)` jobs run in parallel; `release` runs after both succeed and
   publishes the GitHub Release.
   ```bash
   gh run watch              # or: gh run list --workflow=release.yml
   ```

6. **Polish the release notes.** Auto-generated notes list merged PRs/commits.
   Prepend the short install/Gatekeeper/SmartScreen blurb from
   [`end-user-install-notes.md`](end-user-install-notes.md) so first-time
   downloaders aren't scared off by the unsigned warnings.

7. **Verify.** Download one artifact per OS and check its checksum:
   ```bash
   # Windows (PowerShell)
   (Get-FileHash transcribe-setup-vX.Y.Z.exe -Algorithm SHA256).Hash
   # macOS / Linux
   shasum -a 256 -c transcribe-macos-arm64-vX.Y.Z.tar.gz.sha256
   ```

---

## Manual / dry-run builds (no tag)

To test the pipeline without publishing a real version tag: Actions tab →
**Release** → **Run workflow** → enter a `version` (e.g. `v0.11.0-rc1`).
The workflow builds all artifacts and creates a release under that name/tag.
Delete the test release + tag afterward if it was throwaway:

```bash
gh release delete vX.Y.Z-rc1 --cleanup-tag --yes
```

---

## Artifact naming reference

| File | Platform |
|---|---|
| `transcribe-setup-vX.Y.Z.exe` (+ `.sha256`) | Windows installer (per-user, Inno Setup) |
| `transcribe-windows-amd64-vX.Y.Z.zip` (+ `.sha256`) | Windows portable (both exes + README/LICENSE) |
| `transcribe-macos-arm64-vX.Y.Z.tar.gz` (+ `.sha256`) | macOS Apple Silicon |

---

## Troubleshooting

- **Release job fails with HTTP 403 / "Resource not accessible by
  integration".** Workflow permissions are read-only — fix Part A step 2.
- **Windows job: `gcc not found` / CGO error.** The runner's `gcc` (MinGW)
  should be on PATH and `scripts/build-installer.ps1` probes common
  fallbacks. If it regresses, add an explicit toolchain step before the
  build, e.g. `msys2/setup-msys2` (install `mingw-w64-ucrt-x86_64-gcc`) or
  `choco install mingw`, and ensure its `bin` is on PATH.
- **Windows job: `ISCC not found`.** The `choco install innosetup` step
  failed or didn't land on PATH; the build script also probes
  `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`. Re-run the job.
- **macOS job fails compiling the Fyne GUI.** This is the one untested path
  (the unified binary pulls in Fyne on macOS). If it can't link the Cocoa /
  OpenGL frameworks, the fallback is to ship a CLI/TUI-only macOS build
  behind a `nogui` build tag and exclude the `gui` import there. File an
  issue and adjust `cmd/transcribe` before re-tagging.
- **The wrong (old) workflow ran on a tag.** GitHub runs the workflow file
  *as it existed at the tagged commit*. Make sure the tag is on a commit
  that already contains the updated `.github/workflows/release.yml`.

---

## Adding code signing later (not done yet)

When you obtain certificates, the changes are: store secrets in Part A
step 3, add a signing step to the Windows job (e.g. `signtool` /
`AzureSignTool` against the cert) and an `codesign` + `xcrun notarytool`
step to the macOS jobs, then drop the unsigned warnings from the release
notes and README.
