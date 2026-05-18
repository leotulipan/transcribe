package services

import (
	"encoding/json"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// mergeChunks concatenates per-chunk Results, offsetting timestamps by each
// chunk's StartOffset. RawJSON of multi-chunk results is emitted as a JSON
// array of the per-chunk raw payloads.
func mergeChunks(parts []*domain.Result, chunks []domain.Chunk) (*domain.Result, error) {
	if len(parts) == 0 {
		return nil, nil
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	base := *parts[0]
	base.Text = ""
	base.Words = nil
	base.Segments = nil

	var rawAll []json.RawMessage
	for i, p := range parts {
		off := chunks[i].StartOffset
		if i > 0 {
			base.Text += " "
		}
		base.Text += p.Text
		for _, w := range p.Words {
			base.Words = append(base.Words, domain.Word{
				Text: w.Text, Confidence: w.Confidence,
				Start: w.Start + off, End: w.End + off,
			})
		}
		for _, s := range p.Segments {
			base.Segments = append(base.Segments, domain.Segment{
				Text: s.Text, SpeakerID: s.SpeakerID,
				Start: s.Start + off, End: s.End + off,
			})
		}
		rawAll = append(rawAll, json.RawMessage(p.RawJSON))
	}
	bs, err := json.Marshal(rawAll)
	if err != nil {
		return nil, err
	}
	base.RawJSON = bs
	return &base, nil
}
