# Python CLI/TUI Project Blueprint

This document outlines the standard architecture, build system, and patterns for creating robust Command Line Interfaces (CLI) and Text User Interfaces (TUI) in Python. It serves as a template for future tools.

## 1. Installation & Environment Management

We use **UV** (an extremely fast Python package installer and resolver) for managing our Python environment and dependencies. This ensures reproducible and speedy setups across different machines.

### Standard Setup Workflow

1.  **Install UV**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Initialize Virtual Environment**:
    ```bash
    uv venv
    ```

3.  **Install Dependencies**:
    ```bash
    uv pip install -e .
    ```
    This installs the project in editable mode along with all dependencies defined in `pyproject.toml`.

## 2. Build System

The build system is designed to create standalone executables for distribution, targeting Windows x64 (or other platforms as needed) without requiring the end-user to have Python installed.

*   **Build Backend**: `setuptools` is used as the standard build backend in `pyproject.toml`.
*   **Executable Generation**: **PyInstaller** bundles the application into a single executable file.
*   **Automation**: A custom `build.py` script should be used to automate the PyInstaller process. It handles:
    *   **Hidden Imports**: Automatically including dynamic imports that PyInstaller might miss (common with some API libraries).
    *   **Asset Inclusion**: Bundling non-code files (config templates, icons).
    *   **Naming Conventions**: enforcing platform-specific naming (e.g., `app-name-windows-amd64.exe`).

## 3. Technology Stack

We leverage a specific set of modern libraries to ensure a consistent, high-quality user experience.

### Core Libraries
*   **Click**: The backbone of the CLI. It handles argument parsing, flags, subcommands, and help generation.
*   **Rich**: For visual polish. It is used for:
    *   Colored and styled console output.
    *   Layouts (Panels, Columns).
    *   Progress bars and spinners for long-running tasks.
*   **Questionary**: For interactive user input. It powers wizards and menus, allowing users to select options via keyboard navigation rather than typing raw text.

### Reusable Component Patterns
These generic components should be implemented in every tool to ensure consistency:

*   **ConfigManager**: A centralized class for handling configuration. It should:
    *   Load from a persistent JSON/YAML file.
    *   Override with environment variables.
    *   Provide typed accessors for settings (e.g., `get_api_key()`).
*   **ParameterAdapter**: A utility class to normalize input. It translates generic user intents (e.g., "high quality") into specific parameters required by different backend providers or APIs.
*   **UnifiedResult**: A standardized data class (Pydantic model or dataclass) that holds the output of the tool's core operation. This ensures that regardless of which backend performs the work, the rest of the application consumes data in a consistent format.

## 4. Architecture & Design Patterns

We use specific patterns to maintain modularity, testability, and extensibility.

### The Provider Pattern (Classes)
**Usage**: Use abstract base classes to define the "contract" for a core feature that has multiple implementations (e.g., different AI providers, different storage backends).

*   **Base Class**: `BaseProvider` (Abstract). Defines methods like `process()`, `validate_credentials()`, and `get_capabilities()`.
*   **Concrete Implementations**: Classes like `ProviderA`, `ProviderB` inherit from `BaseProvider`. They contain the specific API calls and logic for that service.

### The Mixin Pattern (Shared Capabilities)
**Usage**: Use Mixins to inject specific capabilities into classes without creating a rigid inheritance tree. This is useful for optional features that only *some* providers support.

*   **Example**: `BatchProcessingMixin`
    *   **Purpose**: Adds logic for splitting a large task into smaller chunks and merging results.
    *   **Application**: A provider that doesn't support large payloads natively can inherit this mixin to gain that capability client-side.

### Project Structure
A standard project structure should look like this:

```text
project_root/
├── app_package/           # Main source code
│   ├── cli.py             # Entry point (Click commands)
│   ├── tui/               # Interactive wizards (Questionary/Rich)
│   ├── utils/
│   │   ├── api/           # Provider implementations
│   │   │   ├── base.py    # Abstract Base Class
│   │   │   └── ...
│   │   ├── config.py      # ConfigManager
│   │   └── adapters.py    # ParameterAdapter
├── build.py               # PyInstaller automation
├── pyproject.toml         # Dependencies and metadata
└── README.md
```
