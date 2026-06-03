package format

import "testing"

func TestSpeakerPrefix(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"A", "[Speaker A]: "},
		{"B", "[Speaker B]: "},
		{"0", "[Speaker 0]: "},
		{"12", "[Speaker 12]: "},
		{"Julia", "[Julia]: "},
		{"Gast", "[Gast]: "},
		{"Speaker 1", "[Speaker 1]: "}, // already-formatted-ish named token stays bracketed as-is name
	}
	for _, c := range cases {
		if got := speakerPrefix(c.in); got != c.want {
			t.Errorf("speakerPrefix(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
