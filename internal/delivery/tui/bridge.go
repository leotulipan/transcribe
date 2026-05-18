package tui

import (
	tea "github.com/charmbracelet/bubbletea"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

type progressMsg domain.ProgressEvent
type doneMsg struct {
	result *domain.Result
	err    error
}
type startedMsg struct{ job ports.Job }

// listenProgress returns a tea.Cmd that reads one event from Job.Progress()
// and surfaces it as a progressMsg. The progress screen re-issues this Cmd
// in its Update until the channel closes (signaled by a closed channel).
func listenProgress(job ports.Job) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-job.Progress()
		if !ok {
			res, err := job.Wait()
			return doneMsg{result: res, err: err}
		}
		return progressMsg(ev)
	}
}
