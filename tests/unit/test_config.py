"""
Unit tests for configuration management.

Tests ConfigManager initialization, API key loading/setting,
config directory paths, and error handling.
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from audio_transcribe.utils.config import ConfigManager


@pytest.mark.unit
class TestConfigManagerInitialization:
    """Test ConfigManager initialization and setup."""

    def test_initialization_creates_config_dir(self, tmp_path):
        """Test that initialization creates config directory."""
        with patch('os.getenv') as mock_getenv:
            # Mock LOCALAPPDATA to point to temp directory
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Config directory should be created
            expected_dir = tmp_path / "audio_transcribe"
            assert expected_dir.exists()
            assert config.config_dir == expected_dir

    def test_initialization_on_windows(self, tmp_path):
        """Test initialization on Windows platform."""
        with patch('os.name', 'nt'):
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = str(tmp_path)

                config = ConfigManager()

                assert config.config_dir == tmp_path / "audio_transcribe"
                assert config.app_name == "audio_transcribe"

    def test_initialization_on_unix(self, tmp_path):
        """Test initialization on Unix-like platforms."""
        with patch('os.name', 'posix'):
            with patch.object(Path, 'home') as mock_home:
                mock_home.return_value = tmp_path

                config = ConfigManager()

                assert config.config_dir == tmp_path / ".audio_transcribe"

    def test_config_file_paths(self, tmp_path):
        """Test that config file paths are set correctly."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            assert config.config_file == config.config_dir / "config.json"
            assert config.env_file == config.config_dir / ".env"

    def test_initialization_loads_existing_config(self, tmp_path):
        """Test that initialization loads existing config file."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            # Create existing config
            config_dir = tmp_path / "audio_transcribe"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"

            test_config = {"test_key": "test_value"}
            config_file.write_text(json.dumps(test_config), encoding='utf-8')

            # Initialize ConfigManager
            config = ConfigManager()

            # Should load existing config
            assert config.config == test_config


@pytest.mark.unit
class TestAPIKeyManagement:
    """Test API key getting and setting."""

    def test_set_api_key_updates_env_file(self, tmp_path):
        """Test that setting API key updates .env file."""
        with patch('os.getenv') as mock_getenv:
            # Return tmp_path for LOCALAPPDATA, None for others
            mock_getenv.side_effect = lambda x: str(tmp_path) if x == 'LOCALAPPDATA' else None

            config = ConfigManager()
            config.set_api_key("test_api", "test_key_123")

            # Check .env file was created/updated
            env_file = config.config_dir / ".env"
            assert env_file.exists()

            content = env_file.read_text(encoding='utf-8')
            # dotenv adds quotes around values
            assert "TEST_API_API_KEY" in content and "test_key_123" in content

    def test_set_api_key_updates_environment(self, tmp_path):
        """Test that setting API key updates environment variable."""
        with patch('os.getenv') as mock_getenv:
            # Return tmp_path for LOCALAPPDATA, None for others
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                # For API keys, check os.environ directly
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            config = ConfigManager()
            config.set_api_key("openai", "sk-test123")

            # Environment variable should be set in os.environ
            assert os.environ.get("OPENAI_API_KEY") == "sk-test123"

    def test_get_api_key_from_environment(self, tmp_path):
        """Test getting API key from environment."""
        with patch('os.getenv') as mock_getenv:
            # Return tmp_path for LOCALAPPDATA, check os.environ for API keys
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            config = ConfigManager()

            # Set environment variable directly in os.environ
            os.environ["GROQ_API_KEY"] = "grok_test123"

            # Should retrieve from environment
            key = config.get_api_key("groq")
            assert key == "grok_test123"

            # Clean up
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]

    def test_get_missing_api_key_returns_none(self, tmp_path):
        """Test that getting missing API key returns None."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            config = ConfigManager()

            # Try to get non-existent key
            key = config.get_api_key("nonexistent")
            assert key is None

    def test_multiple_api_keys(self, tmp_path):
        """Test setting and getting multiple API keys."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            config = ConfigManager()

            # Set multiple keys
            config.set_api_key("openai", "sk-openai123")
            config.set_api_key("groq", "gsk-groq456")
            config.set_api_key("assemblyai", "aa-assembly789")

            # Retrieve them from os.environ
            assert os.environ.get("OPENAI_API_KEY") == "sk-openai123"
            assert os.environ.get("GROQ_API_KEY") == "gsk-groq456"
            assert os.environ.get("ASSEMBLYAI_API_KEY") == "aa-assembly789"

            # Clean up
            for api_key in ["OPENAI_API_KEY", "GROQ_API_KEY", "ASSEMBLYAI_API_KEY"]:
                if api_key in os.environ:
                    del os.environ[api_key]


@pytest.mark.unit
class TestConfigDirectoryPaths:
    """Test config directory path handling."""

    def test_config_dir_creation_if_missing(self, tmp_path):
        """Test that config directory is created if missing."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            # Remove config dir if it exists
            config_dir = tmp_path / "audio_transcribe"
            if config_dir.exists():
                config_dir.rmdir()

            # Initialize ConfigManager
            config = ConfigManager()

            # Directory should be created
            assert config_dir.exists()

    def test_config_dir_exists_if_present(self, tmp_path):
        """Test that existing config directory is used."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            # Create config dir
            config_dir = tmp_path / "audio_transcribe"
            config_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ConfigManager
            config = ConfigManager()

            # Should use existing directory
            assert config.config_dir == config_dir

    def test_env_file_path_correct(self, tmp_path):
        """Test that .env file path is correct."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            assert config.env_file.name == ".env"
            assert config.env_file.parent == config.config_dir

    def test_config_json_path_correct(self, tmp_path):
        """Test that config.json path is correct."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            assert config.config_file.name == "config.json"
            assert config.config_file.parent == config.config_dir


@pytest.mark.unit
class TestConfigValueManagement:
    """Test getting and setting config values."""

    def test_get_config_value_default(self, tmp_path):
        """Test getting config value with default."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Get non-existent value with default
            result = config.get("nonexistent_key", "default_value")
            assert result == "default_value"

    def test_set_and_get_config_value(self, tmp_path):
        """Test setting and getting config value."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Set value
            config.set("test_key", "test_value")

            # Get value
            result = config.get("test_key")
            assert result == "test_value"

    def test_set_config_saves_to_file(self, tmp_path):
        """Test that setting config value saves to file."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()
            config.set("my_key", "my_value")

            # Read config file
            config_file = config.config_dir / "config.json"
            assert config_file.exists()

            data = json.loads(config_file.read_text(encoding='utf-8'))
            assert data.get("my_key") == "my_value"

    def test_update_existing_config_value(self, tmp_path):
        """Test updating existing config value."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Set initial value
            config.set("key1", "value1")
            assert config.get("key1") == "value1"

            # Update value
            config.set("key1", "value2")
            assert config.get("key1") == "value2"

    def test_multiple_config_values(self, tmp_path):
        """Test storing multiple config values."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Set multiple values
            config.set("key1", "value1")
            config.set("key2", "value2")
            config.set("key3", "value3")

            # Retrieve all
            assert config.get("key1") == "value1"
            assert config.get("key2") == "value2"
            assert config.get("key3") == "value3"


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in ConfigManager."""

    def test_corrupted_config_json(self, tmp_path):
        """Test handling of corrupted config.json file."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            # Create corrupted config file
            config_dir = tmp_path / "audio_transcribe"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"

            # Write invalid JSON
            config_file.write_text("{invalid json}", encoding='utf-8')

            # Initialize ConfigManager - should not crash
            config = ConfigManager()

            # Should have empty config as fallback
            assert config.config == {}

    def test_save_config_permission_error(self, tmp_path):
        """Test handling of permission error when saving config."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Mock write to raise permission error
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                # Should not crash, just log error
                config.set("test_key", "test_value")

            # Config should still be updated in memory
            assert config.get("test_key") == "test_value"

    def test_load_config_permission_error(self, tmp_path):
        """Test handling of permission error when loading config."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            config = ConfigManager()

            # Mock read to raise permission error
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                # Should not crash, return empty config
                loaded = config._load_config()

            assert loaded == {}

    def test_env_file_write_error(self, tmp_path):
        """Test handling of error when writing to .env file."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            config = ConfigManager()

            # Mock set_key to raise error
            with patch('audio_transcribe.utils.config.set_key', side_effect=Exception("Write error")):
                # Should not crash, just log error
                config.set_api_key("test", "test_key")

            # Environment variable should still be set in os.environ
            assert os.environ.get("TEST_API_KEY") == "test_key"

            # Clean up
            if "TEST_API_KEY" in os.environ:
                del os.environ["TEST_API_KEY"]


@pytest.mark.unit
class TestEnvFileLoading:
    """Test .env file loading behavior."""

    def test_loads_central_env_file(self, tmp_path):
        """Test that central .env file is loaded."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                # Return the actual value from os.environ for API keys
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            # Create .env file with test key
            config_dir = tmp_path / "audio_transcribe"
            config_dir.mkdir(parents=True, exist_ok=True)
            env_file = config_dir / ".env"

            env_file.write_text("TEST_API_KEY=test_value\n", encoding='utf-8')

            # Initialize ConfigManager
            config = ConfigManager()

            # Should have loaded from .env into os.environ
            assert os.environ.get("TEST_API_KEY") == "test_value"

            # Clean up
            if "TEST_API_KEY" in os.environ:
                del os.environ["TEST_API_KEY"]

    def test_loads_local_env_override(self, tmp_path):
        """Test that local .env file is loaded after central .env."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            # Create central .env
            config_dir = tmp_path / "audio_transcribe"
            config_dir.mkdir(parents=True, exist_ok=True)
            central_env = config_dir / ".env"
            central_env.write_text("CENTRAL_KEY=central_value\n", encoding='utf-8')

            # Change to temp directory first
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)

                # Create local .env with different key
                local_env = tmp_path / ".env"
                local_env.write_text("LOCAL_KEY=local_value\n", encoding='utf-8')

                # Initialize ConfigManager
                config = ConfigManager()

                # Both should be loaded
                assert os.environ.get("CENTRAL_KEY") == "central_value"
                assert os.environ.get("LOCAL_KEY") == "local_value"

                # Clean up
                if "CENTRAL_KEY" in os.environ:
                    del os.environ["CENTRAL_KEY"]
                if "LOCAL_KEY" in os.environ:
                    del os.environ["LOCAL_KEY"]
            finally:
                os.chdir(original_cwd)

    def test_missing_env_file_no_error(self, tmp_path):
        """Test that missing .env file doesn't cause error."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = str(tmp_path)

            # Don't create .env file

            # Should not crash
            config = ConfigManager()

            # Config should still be initialized
            assert config.config_dir.exists()


@pytest.mark.unit
class TestBackwardsCompatibility:
    """Test backwards compatibility with old config locations."""

    def test_loads_from_old_transcribe_dir(self, tmp_path):
        """Test loading from old .transcribe directory."""
        with patch('os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'LOCALAPPDATA':
                    return str(tmp_path)
                return os.environ.get(key, default)

            mock_getenv.side_effect = getenv_side_effect

            # Create old config location
            old_dir = tmp_path / ".transcribe"
            old_dir.mkdir(parents=True, exist_ok=True)
            old_env = old_dir / ".env"
            old_env.write_text("OLD_KEY=old_value\n", encoding='utf-8')

            # Mock Path.home() to return tmp_path
            with patch.object(Path, 'home') as mock_home:
                mock_home.return_value = tmp_path

                # Initialize ConfigManager
                config = ConfigManager()

                # Should load from old location into os.environ
                assert os.environ.get("OLD_KEY") == "old_value"

                # Clean up
                if "OLD_KEY" in os.environ:
                    del os.environ["OLD_KEY"]
