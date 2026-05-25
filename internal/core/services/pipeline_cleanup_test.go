package services

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCleanupEmptyWorkDir_RemovesJobDirAndParent(t *testing.T) {
	root := t.TempDir()
	// Simulate the layout that resolveWorkDir creates: <root>/.transcribe-tmp/<base>/
	parent := filepath.Join(root, ".transcribe-tmp")
	jobDir := filepath.Join(parent, "talk")
	require.NoError(t, os.MkdirAll(jobDir, 0o755))

	cleanupEmptyWorkDir(jobDir)

	require.NoDirExists(t, jobDir, "job dir should have been removed")
	require.NoDirExists(t, parent, "parent .transcribe-tmp should have been removed when empty")
}

func TestCleanupEmptyWorkDir_LeavesParentWhenOtherJobsPresent(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, ".transcribe-tmp")
	jobDir := filepath.Join(parent, "talk")
	otherJob := filepath.Join(parent, "other-talk")
	require.NoError(t, os.MkdirAll(jobDir, 0o755))
	require.NoError(t, os.MkdirAll(otherJob, 0o755))

	cleanupEmptyWorkDir(jobDir)

	require.NoDirExists(t, jobDir, "completed job dir should be removed")
	require.DirExists(t, parent, "parent should survive because other-talk still exists")
	require.DirExists(t, otherJob, "sibling job dir must not be touched")
}

func TestCleanupEmptyWorkDir_NonEmptyJobDirIsLeftAlone(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, ".transcribe-tmp")
	jobDir := filepath.Join(parent, "talk")
	require.NoError(t, os.MkdirAll(jobDir, 0o755))
	// Leave a file in the job dir (e.g. a kept intermediate from a transient error)
	require.NoError(t, os.WriteFile(filepath.Join(jobDir, "chunk.mp3"), []byte("x"), 0o644))

	cleanupEmptyWorkDir(jobDir)

	require.DirExists(t, jobDir, "non-empty job dir must not be removed")
}

func TestCleanupEmptyWorkDir_ToleratesMissingDir(t *testing.T) {
	root := t.TempDir()
	jobDir := filepath.Join(root, ".transcribe-tmp", "talk")
	// jobDir was never created — must not panic or return an error
	cleanupEmptyWorkDir(jobDir) // no assertion needed; must not panic
}

func TestCleanupEmptyWorkDir_LeavesNonTranscribeParent(t *testing.T) {
	// Parent is NOT named ".transcribe-tmp" (e.g. fallback path under os.TempDir()).
	// The job dir should be removed, but the parent must not be touched.
	root := t.TempDir()
	parent := filepath.Join(root, "transcribe-talk") // fallback layout, no .transcribe-tmp
	jobDir := filepath.Join(parent, "work")
	require.NoError(t, os.MkdirAll(jobDir, 0o755))

	cleanupEmptyWorkDir(jobDir)

	require.NoDirExists(t, jobDir, "empty job dir should still be removed")
	require.DirExists(t, parent, "non-.transcribe-tmp parent must not be removed")
}

func TestResolveWorkDir_CreatesAndCleanupRemoves(t *testing.T) {
	root := t.TempDir()
	inputPath := filepath.Join(root, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("x"), 0o644))

	workDir, sideBySide := resolveWorkDir(inputPath)
	require.True(t, sideBySide, "should use side-by-side path in writable dir")
	require.DirExists(t, workDir)

	// Simulate successful cleanup: job dir is now empty, run cleanup.
	cleanupEmptyWorkDir(workDir)

	parent := filepath.Join(root, ".transcribe-tmp")
	require.NoDirExists(t, workDir)
	require.NoDirExists(t, parent)
}
