# Code Refactoring & Regression Fix Plan

## Executive Summary

After analyzing the codebase post-refactoring, several critical issues have been identified that impact maintainability, code quality, and introduce regression risks. This plan addresses code duplication, missing abstractions, inconsistent parameter handling, and structural issues.

## Critical Issues Identified

### 1. Code Duplication (High Priority)

#### 1.1 Duplicate Directory Structure
**Problem:** Two parallel directory structures exist:
- `utils/` (old) vs `audio_transcribe/utils/` (new)
- `transcribe_helpers/` (old) vs `audio_transcribe/transcribe_helpers/` (new)
- Old monolithic `utils/transcription_api.py` still exists

**Impact:** 
- Confusion about which files are actually used
- Risk of editing wrong files
- Import inconsistencies
- Increased maintenance burden

**Files Affected:**
- `utils/api/base.py` vs `audio_transcribe/utils/api/base.py` (different implementations)
- `utils/parsers.py` vs `audio_transcribe/utils/parsers.py`
- `transcribe_helpers/audio_processing.py` vs `audio_transcribe/transcribe_helpers/audio_processing.py`
- `transcribe_helpers/output_formatters.py` vs `audio_transcribe/transcribe_helpers/output_formatters.py`

**Solution:**
1. Audit all imports to determine which structure is actually used
2. Remove unused `utils/` and `transcribe_helpers/` directories (keep only `audio_transcribe/` versions)
3. Update any remaining imports
4. Delete `utils/transcription_api.py` if it's not referenced

#### 1.2 Duplicate Base Class Implementations
**Problem:** `audio_transcribe/utils/api/base.py` and `utils/api/base.py` have different method signatures:
- Different `mask_api_key()` implementations
- Different `with_retry()` signatures
- Different `save_result()` implementations
- Missing `load_from_env()` in one version

**Solution:**
1. Consolidate to single base class with all methods
2. Ensure consistent method signatures across all API implementations
3. Add missing utility methods to base class

### 2. Not DRY (Repeated Code Fragments)

#### 2.1 Model Defaults Hardcoded in Multiple Places
**Problem:** Default models are defined in:
- `audio_transcribe/cli.py` (lines 228-239): Hardcoded defaults per API
- `audio_transcribe/utils/models.py`: MODEL_REGISTRY with defaults
- Individual API classes: Some have defaults, some don't
- TUI interactive mode: Uses models.py but CLI doesn't consistently use it

**Example:**
```python
# cli.py line 232-236
if api_name == "groq":
    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-large-v3')
elif api_name == "openai":
    transcribe_kwargs['model'] = kwargs.get('model', 'whisper-1')
elif api_name == "assemblyai":
    transcribe_kwargs['model'] = kwargs.get('model', 'best')
```

**Solution:**
1. Create centralized `get_default_model(api_name)` function in `models.py`
2. Use this function consistently in `cli.py`, API classes, and TUI
3. Remove all hardcoded defaults from `cli.py`
4. Ensure API classes use `models.py` defaults when model is None

#### 2.2 Audio Conversion Logic Duplicated
**Problem:** Audio optimization/conversion happens in:
- `cli.py` (line 215): Calls `optimize_audio_for_api()`
- Individual API classes: Some still do their own conversion (e.g., `openai.py` line 191, `elevenlabs.py` line 114)
- `assemblyai.py`: Comment says optimization handled in cli.py but still has extraction code

**Solution:**
1. Ensure ALL APIs rely on `optimize_audio_for_api()` from `cli.py`
2. Remove duplicate conversion logic from API classes
3. Add clear documentation that APIs receive pre-optimized files
4. Create abstraction layer: `AudioPreprocessor` class that handles all conversion

#### 2.3 File Size Checking Duplicated
**Problem:** File size checks occur in:
- `cli.py`: Before calling API
- Individual API classes: `elevenlabs.py` (line 128), `openai.py` (line 208)
- `audio_processing.py`: `check_file_size()` and `get_api_file_size_limit()`

**Solution:**
1. Centralize file size validation in `optimize_audio_for_api()`
2. Remove redundant checks from API classes
3. APIs should trust that files passed to them are within limits

#### 2.4 Parameter Name Inconsistencies
**Problem:** Different APIs use different parameter names:
- ElevenLabs: `model_id` vs others: `model`
- ElevenLabs: `language_code` vs others: `language`
- Some APIs: `speaker_labels`, others: `diarize`

**Solution:**
1. Create `ParameterNormalizer` class to map CLI params to API-specific params
2. Normalize all parameters before passing to API classes
3. Document parameter mapping in base class docstrings

### 3. Missing Proper Abstractions

#### 3.1 No Unified Audio Processing Pipeline
**Problem:** Audio processing is scattered:
- `cli.py`: Calls `optimize_audio_for_api()`
- Individual APIs: Some do extraction, some don't
- No clear contract about what format APIs receive

**Solution:**
1. Create `AudioPreprocessor` class:
   ```python
   class AudioPreprocessor:
       def prepare_for_api(self, file_path, api_name, **options) -> PreparedAudio:
           # Returns PreparedAudio with path, format, size info
   ```
2. All APIs receive `PreparedAudio` objects
3. Centralize all conversion logic in this class

#### 3.2 No Unified Parameter Handling
**Problem:** Parameters are handled differently:
- CLI collects params
- `cli.py` transforms some params (model defaults)
- APIs receive kwargs and extract what they need
- No validation layer

**Solution:**
1. Create `TranscriptionParameters` dataclass:
   ```python
   @dataclass
   class TranscriptionParameters:
       api_name: str
       model: Optional[str]
       language: Optional[str]
       # ... all params
       
       def normalize_for_api(self, api_name: str) -> Dict[str, Any]:
           # Returns API-specific parameter dict
   ```
2. Use this throughout the codebase
3. Add validation in the dataclass

#### 3.3 JSON to SRT Conversion Not Abstracted Enough
**Problem:** SRT generation logic:
- Lives in `transcribe_helpers/output_formatters.py`
- Called from `utils/formatters.py` which wraps it
- Different modes (standard, word, davinci) handled with if/else
- Complex parameter passing through multiple layers

**Solution:**
1. Create `SRTFormatter` class hierarchy:
   ```python
   class SRTFormatter(ABC):
       @abstractmethod
       def format(self, result: TranscriptionResult, **options) -> str
   
   class StandardSRTFormatter(SRTFormatter): ...
   class WordSRTFormatter(SRTFormatter): ...
   class DaVinciSRTFormatter(SRTFormatter): ...
   ```
2. Use factory pattern to select formatter
3. Simplify parameter passing

### 4. Inconsistent Defaults

#### 4.1 Default Values Scattered
**Problem:** Defaults defined in:
- CLI option defaults (click decorators)
- `cli.py` logic (DaVinci mode defaults)
- `models.py` registry
- Individual API classes
- ConfigManager

**Solution:**
1. Create `Defaults` class in `audio_transcribe/utils/defaults.py`:
   ```python
   class Defaults:
       API = "groq"
       MODEL = {  # Per-API defaults
           "groq": "whisper-large-v3",
           "openai": "whisper-1",
           ...
       }
       OUTPUT_FORMATS = ["text", "srt"]
       CHARS_PER_LINE = 80
       DAVINCI_CHARS_PER_LINE = 500
       # ... all defaults
   ```
2. Reference this class everywhere
3. Allow ConfigManager to override defaults

#### 4.2 DaVinci Mode Defaults Applied Inconsistently
**Problem:** `cli.py` lines 522-527 apply DaVinci defaults, but:
- Only applies if `davinci_srt` flag is set
- Doesn't prevent override if user explicitly sets values
- Logic is buried in main() function

**Solution:**
1. Move DaVinci defaults to `Defaults` class
2. Apply in `TranscriptionParameters.normalize_for_api()`
3. Document that explicit CLI args override defaults

### 5. Parameter Handling Issues

#### 5.1 Model Parameter Inconsistencies
**Problem:**
- ElevenLabs uses `model_id` parameter
- Other APIs use `model`
- `cli.py` transforms `model` to `model_id` for ElevenLabs (line 229)
- But ElevenLabs API class signature uses `model_id` (line 84)

**Solution:**
1. Standardize all APIs to use `model` parameter
2. Update ElevenLabs API to accept `model` and map internally
3. Remove transformation logic from `cli.py`

#### 5.2 Language Parameter Inconsistencies
**Problem:**
- Some APIs: `language`
- ElevenLabs: `language_code`
- AssemblyAI: `language_code` when set, `language_detection` when not

**Solution:**
1. Standardize CLI to always use `language`
2. Create parameter mapper in base class or ParameterNormalizer
3. Map `language` -> `language_code` for APIs that need it

#### 5.3 Missing Parameter Validation
**Problem:** No validation for:
- Model names (check against available models)
- Language codes (format validation)
- File paths (existence, format)
- Numeric ranges (chunk_length, overlap, etc.)

**Solution:**
1. Add validation to `TranscriptionParameters` dataclass
2. Validate before API calls
3. Provide clear error messages

### 6. Structural Issues

#### 6.1 Circular Import Risks
**Problem:** Potential circular imports:
- `cli.py` imports from `utils.api`
- `utils.api` imports from `transcribe_helpers`
- `transcribe_helpers` might import from `utils`

**Solution:**
1. Audit all imports
2. Use lazy imports where necessary
3. Ensure clear dependency hierarchy: `cli` -> `utils` -> `transcribe_helpers`

#### 6.2 Missing Type Hints
**Problem:** Many functions lack proper type hints, making it hard to:
- Understand function contracts
- Catch errors early
- Use IDE autocomplete

**Solution:**
1. Add type hints to all public functions
2. Use `TypedDict` for complex dictionaries
3. Add return type annotations

#### 6.3 Error Handling Inconsistencies
**Problem:** Error handling varies:
- Some APIs raise exceptions
- Some return None/empty results
- Some log and continue
- Retry logic differs

**Solution:**
1. Standardize error handling in base class
2. Define custom exception hierarchy:
   ```python
   class TranscriptionError(Exception): ...
   class APIKeyError(TranscriptionError): ...
   class FileSizeError(TranscriptionError): ...
   class APIError(TranscriptionError): ...
   ```
3. Use consistent error handling patterns

## Implementation Plan

### Phase 1: Cleanup & Consolidation (Critical)
**Goal:** Remove duplication and establish single source of truth

1. **Audit and Remove Duplicate Directories**
   - [ ] Identify which `utils/` and `transcribe_helpers/` directories are actually used
   - [ ] Check all imports across codebase
   - [ ] Remove unused directories
   - [ ] Update any remaining imports

2. **Consolidate Base Class**
   - [ ] Merge `utils/api/base.py` and `audio_transcribe/utils/api/base.py`
   - [ ] Ensure all methods are present
   - [ ] Update all API classes to use consolidated base
   - [ ] Remove old base class

3. **Remove Old Monolithic File**
   - [ ] Check if `utils/transcription_api.py` is referenced
   - [ ] Remove if unused
   - [ ] Update imports if still needed

### Phase 2: Centralize Defaults & Configuration
**Goal:** Single source for all default values

1. **Create Defaults Module**
   - [ ] Create `audio_transcribe/utils/defaults.py`
   - [ ] Move all default values here
   - [ ] Include per-API model defaults
   - [ ] Include DaVinci mode defaults

2. **Update Models Registry**
   - [ ] Ensure `models.py` uses `defaults.py`
   - [ ] Add `get_default_model()` function
   - [ ] Ensure it's used everywhere

3. **Update CLI to Use Centralized Defaults**
   - [ ] Remove hardcoded defaults from `cli.py`
   - [ ] Use `get_default_model()` for model defaults
   - [ ] Use `Defaults` class for other defaults
   - [ ] Apply DaVinci defaults through `Defaults` class

### Phase 3: Create Proper Abstractions
**Goal:** Abstract common operations into reusable classes

1. **Create AudioPreprocessor Class**
   - [ ] Create `audio_transcribe/utils/audio_preprocessor.py`
   - [ ] Move `optimize_audio_for_api()` logic here
   - [ ] Create `PreparedAudio` dataclass
   - [ ] Update `cli.py` to use `AudioPreprocessor`
   - [ ] Remove audio conversion from API classes

2. **Create Parameter Normalization**
   - [ ] Create `audio_transcribe/utils/parameters.py`
   - [ ] Create `TranscriptionParameters` dataclass
   - [ ] Add `normalize_for_api()` method
   - [ ] Update `cli.py` to use this class
   - [ ] Update API classes to expect normalized parameters

3. **Create SRT Formatter Hierarchy**
   - [ ] Create `audio_transcribe/utils/formatters/srt_formatters.py`
   - [ ] Create base `SRTFormatter` class
   - [ ] Create `StandardSRTFormatter`, `WordSRTFormatter`, `DaVinciSRTFormatter`
   - [ ] Create factory function
   - [ ] Update `formatters.py` to use new hierarchy

### Phase 4: Standardize Parameter Handling
**Goal:** Consistent parameter names and handling across all APIs

1. **Standardize Model Parameter**
   - [ ] Update ElevenLabs API to accept `model` instead of `model_id`
   - [ ] Remove `model_id` transformation from `cli.py`
   - [ ] Ensure all APIs use `model` consistently

2. **Standardize Language Parameter**
   - [ ] Create language parameter mapper
   - [ ] Map `language` -> `language_code` for APIs that need it
   - [ ] Update all API classes

3. **Add Parameter Validation**
   - [ ] Add validation methods to `TranscriptionParameters`
   - [ ] Validate model names against available models
   - [ ] Validate language codes
   - [ ] Validate numeric ranges

### Phase 5: Improve Error Handling
**Goal:** Consistent error handling across codebase

1. **Create Exception Hierarchy**
   - [ ] Create `audio_transcribe/exceptions.py`
   - [ ] Define custom exception classes
   - [ ] Update API classes to use custom exceptions

2. **Standardize Error Handling**
   - [ ] Update base class error handling
   - [ ] Ensure consistent retry logic
   - [ ] Improve error messages

### Phase 6: Code Quality Improvements
**Goal:** Improve maintainability and developer experience

1. **Add Type Hints**
   - [ ] Add type hints to all public functions
   - [ ] Use `TypedDict` for complex dictionaries
   - [ ] Add return type annotations

2. **Improve Documentation**
   - [ ] Add docstrings to all classes and functions
   - [ ] Document parameter mappings
   - [ ] Document default behaviors

3. **Add Unit Tests**
   - [ ] Test `Defaults` class
   - [ ] Test `ParameterNormalizer`
   - [ ] Test `AudioPreprocessor`
   - [ ] Test SRT formatters

## File Changes Summary

### New Files to Create
- `audio_transcribe/utils/defaults.py` - Centralized defaults
- `audio_transcribe/utils/parameters.py` - Parameter normalization
- `audio_transcribe/utils/audio_preprocessor.py` - Audio processing abstraction
- `audio_transcribe/utils/formatters/srt_formatters.py` - SRT formatter hierarchy
- `audio_transcribe/exceptions.py` - Custom exceptions

### Files to Modify
- `audio_transcribe/cli.py` - Remove hardcoded defaults, use abstractions
- `audio_transcribe/utils/models.py` - Use centralized defaults
- `audio_transcribe/utils/api/base.py` - Consolidate with old base
- `audio_transcribe/utils/api/elevenlabs.py` - Standardize parameters
- `audio_transcribe/utils/api/openai.py` - Remove audio conversion
- `audio_transcribe/utils/api/assemblyai.py` - Remove audio conversion
- `audio_transcribe/utils/api/groq.py` - Remove audio conversion
- `audio_transcribe/utils/formatters.py` - Use new SRT formatters

### Files to Delete
- `utils/` directory (if unused)
- `transcribe_helpers/` directory (if unused)
- `utils/transcription_api.py` (if unused)

## Testing Strategy

1. **Regression Testing**
   - Test all APIs with same inputs as before refactoring
   - Verify output files are identical
   - Test TUI flows

2. **Integration Testing**
   - Test parameter normalization
   - Test audio preprocessing
   - Test SRT formatting

3. **Unit Testing**
   - Test new abstraction classes
   - Test default value resolution
   - Test parameter validation

## Risk Mitigation

1. **Incremental Changes**
   - Make changes in phases
   - Test after each phase
   - Keep old code until new code is verified

2. **Backward Compatibility**
   - Maintain CLI interface
   - Keep file output formats identical
   - Preserve existing behavior

3. **Documentation**
   - Document all changes
   - Update README if needed
   - Add migration notes if breaking changes

## Success Criteria

1. ✅ No duplicate code directories
2. ✅ Single source of truth for defaults
3. ✅ Consistent parameter handling across all APIs
4. ✅ Proper abstractions for audio processing and formatting
5. ✅ All APIs use same base class with consistent methods
6. ✅ No hardcoded defaults in business logic
7. ✅ Clear separation of concerns
8. ✅ Improved testability through abstractions

## Estimated Effort

- **Phase 1 (Cleanup):** 2-3 hours
- **Phase 2 (Defaults):** 2-3 hours
- **Phase 3 (Abstractions):** 4-6 hours
- **Phase 4 (Parameters):** 3-4 hours
- **Phase 5 (Error Handling):** 2-3 hours
- **Phase 6 (Quality):** 3-4 hours

**Total:** 16-23 hours

## Notes

- This refactoring should be done incrementally
- Each phase should be tested before moving to next
- Keep git commits small and focused
- Consider creating a feature branch for this work
- Update TASKS.md as work progresses

