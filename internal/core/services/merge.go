package services

import (
	"sort"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// LabeledTrack pairs a single-speaker transcription Result with the speaker
// label the user assigned to it and an optional time offset used to align the
// track against the others (tracks recorded on separate mics may not share an
// exact zero point).
type LabeledTrack struct {
	Result *domain.Result
	Label  string
	Offset time.Duration
}

// MergeResults interleaves several single-speaker transcripts into one combined
// transcript. Each track's words are shifted by its offset, stamped with the
// track's label as the speaker, then all words are stable-sorted by start time.
// The de-duplicated speaker set and the carried-over language/provider/model
// come from the input tracks (first non-empty wins).
func MergeResults(tracks []LabeledTrack) *domain.Result {
	out := &domain.Result{}
	seen := map[string]bool{}
	var all []domain.Word
	for _, t := range tracks {
		if t.Result == nil {
			continue
		}
		for _, w := range t.Result.Words {
			w.Speaker = t.Label
			w.Start += t.Offset
			w.End += t.Offset
			all = append(all, w)
		}
		if t.Label != "" && !seen[t.Label] {
			seen[t.Label] = true
			out.Speakers = append(out.Speakers, domain.Speaker{ID: t.Label, Label: t.Label})
		}
		if out.Language == "" {
			out.Language = t.Result.Language
		}
		if out.Provider == "" {
			out.Provider = t.Result.Provider
		}
		if out.Model == "" {
			out.Model = t.Result.Model
		}
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].Start < all[j].Start })
	out.Words = all

	// Plain-text fallback: simple space-joined words. Speaker-labelled rendering
	// is produced by the text formatter from Words when SpeakerLabels is set.
	parts := make([]string, len(all))
	for i, w := range all {
		parts[i] = w.Text
	}
	out.Text = strings.Join(parts, " ")
	return out
}
