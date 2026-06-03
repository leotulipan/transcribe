package format

// speakerPrefix renders the inline speaker prefix for a subtitle/text block.
//
// Bare diarization tokens — a single letter ("A") or a numeric id ("0", "12") —
// are rendered as "[Speaker X]: " so anonymous provider diarization reads
// naturally. Named labels supplied by the user (e.g. "Julia", "Gast" from the
// merge workflow) are rendered verbatim as "[Julia]: " without the literal word
// "Speaker". An empty speaker yields no prefix.
func speakerPrefix(speaker string) string {
	if speaker == "" {
		return ""
	}
	if isBareSpeakerToken(speaker) {
		return "[Speaker " + speaker + "]: "
	}
	return "[" + speaker + "]: "
}

// isBareSpeakerToken reports whether s is an anonymous diarization id: either a
// single ASCII letter or an all-digit string.
func isBareSpeakerToken(s string) bool {
	if len(s) == 1 && ((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z')) {
		return true
	}
	allDigits := true
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			allDigits = false
			break
		}
	}
	return allDigits && len(s) > 0
}
