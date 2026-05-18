package services

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestDavinciApply_InsertsPauseAndUppercasesFillers(t *testing.T) {
	res := &domain.Result{
		Words: []domain.Word{
			{Text: "Wir", Start: 1100 * time.Millisecond, End: 1300 * time.Millisecond},
			{Text: "ähm", Start: 1350 * time.Millisecond, End: 1500 * time.Millisecond},
			{Text: "testen", Start: 1600 * time.Millisecond, End: 2000 * time.Millisecond},
			// 2.0s gap (exceeds 1.5s threshold)
			{Text: "Nochmal", Start: 4000 * time.Millisecond, End: 4500 * time.Millisecond},
		},
	}
	applyDavinci(res, &domain.DaVinciOptions{
		SilentPortionThreshold: 1500 * time.Millisecond,
	})

	var texts []string
	for _, w := range res.Words {
		texts = append(texts, w.Text)
	}
	joined := strings.Join(texts, " ")
	require.Contains(t, joined, "ÄHM", "filler must be uppercased")
	require.Contains(t, joined, "(...)", "pause must be inserted")
	// pause start should match previous end, end should match next start
	var pauseIdx int
	for i, w := range res.Words {
		if w.Text == "(...)" {
			pauseIdx = i
		}
	}
	require.Equal(t, 2000*time.Millisecond, res.Words[pauseIdx].Start)
	require.Equal(t, 4000*time.Millisecond, res.Words[pauseIdx].End)
}

func TestDavinciApply_NoOpWhenOptsNil(t *testing.T) {
	res := &domain.Result{Words: []domain.Word{{Text: "hi"}}}
	applyDavinci(res, nil)
	require.Equal(t, "hi", res.Words[0].Text)
}
