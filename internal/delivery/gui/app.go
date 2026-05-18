package gui

import (
	"context"
	"errors"
	"sync"

	"fyne.io/fyne/v2/app"

	"github.com/leotulipan/transcribe/internal/ports"
)

// Deps holds the GUI's connection to the rest of the program. Service and
// Config are protected by a RWMutex so the Settings window can call Reload
// after a save without restarting the binary. In-flight jobs keep a
// reference to the service pointer they captured at submit time, so the
// old service stays alive until those jobs finish — Reload is safe to
// call any time.
type Deps struct {
	Logger     ports.Logger
	SaveConfig func(ports.Config) error

	// LoadConfig and BuildService are required for Reload to work. If left
	// nil, Reload returns ErrReloadNotWired.
	LoadConfig   func() (ports.Config, error)
	BuildService func(ports.Config) (ports.TranscribeService, error)

	mu      sync.RWMutex
	service ports.TranscribeService
	config  ports.Config
}

// ErrReloadNotWired is returned by Reload when LoadConfig or BuildService
// hasn't been provided to the Deps struct.
var ErrReloadNotWired = errors.New("gui: Reload requires LoadConfig and BuildService to be wired")

// NewDeps constructs a Deps with the initial service + config already loaded.
// The reload closures are optional but recommended; without them Reload
// is a no-op error.
func NewDeps(svc ports.TranscribeService, cfg ports.Config, log ports.Logger,
	saveConfig func(ports.Config) error,
	loadConfig func() (ports.Config, error),
	buildService func(ports.Config) (ports.TranscribeService, error),
) *Deps {
	return &Deps{
		Logger:       log,
		SaveConfig:   saveConfig,
		LoadConfig:   loadConfig,
		BuildService: buildService,
		service:      svc,
		config:       cfg,
	}
}

// Service returns the current transcribe service. Cheap; takes RLock.
func (d *Deps) Service() ports.TranscribeService {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.service
}

// Config returns the current loaded config snapshot. Cheap; takes RLock.
func (d *Deps) Config() ports.Config {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.config
}

// Reload re-reads the config from disk and rebuilds the service. In-flight
// jobs are unaffected; they keep the old service pointer. New jobs use the
// new service.
func (d *Deps) Reload() error {
	if d.LoadConfig == nil || d.BuildService == nil {
		return ErrReloadNotWired
	}
	cfg, err := d.LoadConfig()
	if err != nil {
		return err
	}
	svc, err := d.BuildService(cfg)
	if err != nil {
		return err
	}
	d.mu.Lock()
	d.config = cfg
	d.service = svc
	d.mu.Unlock()
	return nil
}

// Run blocks until the user closes the window. ctx cancellation closes any
// in-flight job and ends the program loop.
func Run(ctx context.Context, deps *Deps) error {
	a := app.NewWithID(appID)
	win := newMainWindow(a, ctx, deps)
	win.Show()
	a.Run()
	return nil
}
