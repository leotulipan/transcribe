param(
    [Parameter(Mandatory=$true)]
    [string]$DirectoryPath
)

# Ensure the directory path exists
if (-not (Test-Path -Path $DirectoryPath -PathType Container)) {
    Write-Error "The specified directory does not exist: $DirectoryPath"
    exit 1
}

# Get all .txt files except those ending with _cleaned.txt
$textFiles = Get-ChildItem -Path $DirectoryPath -Filter "*.txt" | 
    Where-Object { $_.Name -notlike "*_cleaned.txt" }

if ($textFiles.Count -eq 0) {
    Write-Host "No .txt files found. Running transcription script first..." -ForegroundColor Yellow
    
    # Run the transcription script
    Write-Host "Executing: uv run C:\Users\leona\OneDrive\_2_Areas\Scripts.Transcribe\transcribe.py -v -d --folder $DirectoryPath"
    & uv run C:\Users\leona\OneDrive\_2_Areas\Scripts.Transcribe\transcribe.py -v -d --folder $DirectoryPath
    
    # Check again for .txt files
    $textFiles = Get-ChildItem -Path $DirectoryPath -Filter "*.txt" | 
        Where-Object { $_.Name -notlike "*_cleaned.txt" }
    
    if ($textFiles.Count -eq 0) {
        Write-Host "Still no .txt files found after transcription. Exiting." -ForegroundColor Red
        exit 0
    }
    
    Write-Host "Transcription complete. Continuing with cleaning..." -ForegroundColor Green
}

Write-Host "Found $($textFiles.Count) text files to check..."

# Files to actually process
$filesToProcess = @()

foreach ($file in $textFiles) {
    $outputFileName = $file.FullName -replace '\.txt$', '_cleaned.txt'
    
    # Skip if output file already exists
    if (Test-Path -Path $outputFileName) {
        Write-Host "Skipping $($file.Name) - output file already exists" -ForegroundColor Yellow
        continue
    }
    
    # Skip empty files
    if ((Get-Item $file.FullName).Length -eq 0) {
        Write-Host "Skipping $($file.Name) - file is empty" -ForegroundColor Yellow
        continue
    }
    
    $filesToProcess += $file
}

Write-Host "Processing $($filesToProcess.Count) files..." -ForegroundColor Cyan

foreach ($file in $filesToProcess) {
    $outputFileName = $file.FullName -replace '\.txt$', '_cleaned.txt'
    Write-Host "Processing $($file.Name)..."
    
    try {
        # Using the & operator to safely call external commands with parameters
        $output = & llm -a "$($file.FullName)" -t clean_transcript --no-stream
        
        # Write the output to file with OEM encoding
        $output | Set-Content -Path $outputFileName -Encoding OEM
        
        if (Test-Path -Path $outputFileName) {
            Write-Host "Successfully created $([System.IO.Path]::GetFileName($outputFileName))" -ForegroundColor Green
        } else {
            Write-Host "Failed to create output file for $($file.Name)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "Error processing $($file.Name): $_" -ForegroundColor Red
    }
}

Write-Host "Processing complete!" -ForegroundColor Green
