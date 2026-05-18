package domain

import "testing"

// Language utilities (ISO-639-1 / ISO-639-3 normalization, validation, mapping)
// are not yet ported from Python. These tests lock the contract before the
// feature lands. See docs/plans/2-feature-parity-completion.md (no current phase
// covers this directly — propose adding under a Phase 5 if/when promoted).
//
// Python source: audio_transcribe/transcribe_helpers/language_utils.py
// Python tests:  tests/unit/test_language_utils.py

func TestLanguage_NormalizeISO6391_LowercasesAndValidates(t *testing.T) {
	t.Skip("pending: language utilities not yet ported to Go")
	// expected behavior:
	//   Normalize("EN") == "en"
	//   Normalize("de") == "de"
	//   Normalize("xx") returns ErrUnsupportedLanguage
}

func TestLanguage_ConvertISO6393To6391(t *testing.T) {
	t.Skip("pending: language utilities not yet ported to Go")
	// expected behavior:
	//   ToISO6391("deu") == "de"
	//   ToISO6391("eng") == "en"
	//   ToISO6391("fra") == "fr"
}

func TestLanguage_IsSupported(t *testing.T) {
	t.Skip("pending: language utilities not yet ported to Go")
	// expected behavior:
	//   IsSupported("en") == true
	//   IsSupported("zz") == false
}
