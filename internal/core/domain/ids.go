package domain

type ProviderID string

const (
	ProviderAssemblyAI ProviderID = "assemblyai"
	ProviderElevenLabs ProviderID = "elevenlabs"
	ProviderGroq       ProviderID = "groq"
	ProviderOpenAI     ProviderID = "openai"
	ProviderGemini     ProviderID = "gemini"
	ProviderMistral    ProviderID = "mistral"
)

type OutputFormat string

const (
	FormatText       OutputFormat = "text"
	FormatSRT        OutputFormat = "srt"
	FormatDavinciSRT OutputFormat = "davinci_srt"
)

// NeedsTimestamps reports whether this output format requires word-level timing.
func (f OutputFormat) NeedsTimestamps() bool {
	return f == FormatSRT || f == FormatDavinciSRT
}
