package services

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

func TestMergeResults_InterleavesByStartAndLabels(t *testing.T) {
	julia := &domain.Result{
		Language: "deu",
		Provider: domain.ProviderElevenLabs,
		Words: []domain.Word{
			{Text: "Hallo", Start: 0, End: 1 * time.Second},
			{Text: "Gast", Start: 4 * time.Second, End: 5 * time.Second},
		},
	}
	gast := &domain.Result{
		Words: []domain.Word{
			{Text: "Servus", Start: 2 * time.Second, End: 3 * time.Second},
		},
	}

	out := MergeResults([]LabeledTrack{
		{Result: julia, Label: "Julia"},
		{Result: gast, Label: "Gast"},
	})

	require.Len(t, out.Words, 3)
	require.Equal(t, "Hallo", out.Words[0].Text)
	require.Equal(t, "Julia", out.Words[0].Speaker)
	require.Equal(t, "Servus", out.Words[1].Text)
	require.Equal(t, "Gast", out.Words[1].Speaker)
	require.Equal(t, "Gast", out.Words[2].Text) // the word "Gast" spoken by Julia at 4s
	require.Equal(t, "Julia", out.Words[2].Speaker)

	require.Equal(t, "deu", out.Language)
	require.Equal(t, domain.ProviderElevenLabs, out.Provider)

	ids := []string{out.Speakers[0].ID, out.Speakers[1].ID}
	require.ElementsMatch(t, []string{"Julia", "Gast"}, ids)
}

func TestMergeResults_AppliesOffset(t *testing.T) {
	a := &domain.Result{Words: []domain.Word{{Text: "first", Start: 0, End: 1 * time.Second}}}
	b := &domain.Result{Words: []domain.Word{{Text: "shifted", Start: 0, End: 1 * time.Second}}}

	// b is offset by -500ms; without offset the stable sort would keep a first
	// (equal starts). With the negative offset, b should sort before a.
	out := MergeResults([]LabeledTrack{
		{Result: a, Label: "A"},
		{Result: b, Label: "B", Offset: -500 * time.Millisecond},
	})

	require.Equal(t, "shifted", out.Words[0].Text)
	require.Equal(t, -500*time.Millisecond, out.Words[0].Start)
}
