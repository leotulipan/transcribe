package tui

import (
	"github.com/charmbracelet/bubbles/filepicker"
	tea "github.com/charmbracelet/bubbletea"
)

var audioVideoExts = []string{
	".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm",
	".mov", ".mkv", ".aac", ".aiff",
}

type filePicker struct {
	pre Prefill
	fp  filepicker.Model
}

func newFilePicker(p Prefill) *filePicker {
	fp := filepicker.New()
	fp.AllowedTypes = audioVideoExts
	return &filePicker{pre: p, fp: fp}
}

func (f *filePicker) Init() tea.Cmd { return f.fp.Init() }

func (f *filePicker) Update(msg tea.Msg) (screen, tea.Cmd) {
	var cmd tea.Cmd
	f.fp, cmd = f.fp.Update(msg)
	if selected, path := f.fp.DidSelectFile(msg); selected {
		f.pre.InputPath = path
		return f, func() tea.Msg { return advanceMsg{pre: f.pre} }
	}
	return f, cmd
}

func (f *filePicker) View() string {
	return titleStyle.Render("Pick an audio or video file") + "\n\n" +
		f.fp.View() + "\n" +
		helpStyle.Render("enter: select • esc: cancel")
}
