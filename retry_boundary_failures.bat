@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "RESULT=0"
set "BACKUP_DIR=outputs\before_bilinear_boundary_fix"
set "DATASET_DIR=datasets\water_a1_i300_380_830"

if not "%~3"=="" goto usage
if /I "%~1"=="backup" goto backup
if /I "%~1"=="retry" goto retry
goto usage

:backup
if not "%~2"=="" goto usage
if exist "%BACKUP_DIR%" (
    echo Refusing to overwrite existing backup: %BACKUP_DIR%
    set "RESULT=1"
    goto finish
)
if not exist "%DATASET_DIR%\batch_plan.json" (
    echo Missing existing dataset plan: %DATASET_DIR%\batch_plan.json
    set "RESULT=1"
    goto finish
)
call :require_binaries "build\Release"
if errorlevel 1 (
    set "RESULT=1"
    goto finish
)

rem The destination must be new. A failed backup is preserved for inspection.
mkdir "%BACKUP_DIR%\modules"
if errorlevel 1 goto backup_failed
call :copy_checked "build\Release\rainbow_generate.exe" "%BACKUP_DIR%\rainbow_generate.exe"
if errorlevel 1 goto backup_failed
for %%M in (raindrop_trace.optixir patch_build.fatbin patch_query.optixir patch_optics.fatbin phase_cdf.fatbin) do (
    call :copy_checked "build\Release\modules\%%M" "%BACKUP_DIR%\modules\%%M"
    if errorlevel 1 goto backup_failed
)
call :copy_checked "%DATASET_DIR%\batch_plan.json" "%BACKUP_DIR%\batch_plan.json"
if errorlevel 1 goto backup_failed
if exist "%DATASET_DIR%\generation_history.json" (
    call :copy_checked "%DATASET_DIR%\generation_history.json" "%BACKUP_DIR%\generation_history.json"
    if errorlevel 1 goto backup_failed
)
echo Backup copied and verified byte-for-byte: %BACKUP_DIR%
echo You can now rebuild. Keep this backup unchanged for generation registration.
goto finish

:backup_failed
echo Backup failed. Existing copied files have been preserved: %BACKUP_DIR%
echo Do not rebuild until a complete backup of the current binaries is available.
set "RESULT=1"
goto finish

:retry
call :require_binaries "%BACKUP_DIR%"
if errorlevel 1 (
    set "RESULT=1"
    goto finish
)
rem The fixed dataset has 300 * 91 = 27300 conditions. This default includes
rem every possible failed condition; an explicit second argument limits a smoke run.
set "RETRY_LIMIT=27300"
if not "%~2"=="" set "RETRY_LIMIT=%~2"

rem These scientific settings match gen.bat, including its explicit
rem --allow-underresolved policy. No resolution or acceptance tolerance is changed.
rem The Python runner checks conditions and the previous active binary hashes,
rem preserves successful records, and retries recorded failures once each.
call generate_dataset.bat ^
    --config configs\dataset_water_a1.json ^
    --inclination-count 300 ^
    --wavelength-range 380 830 5 ^
    --out-root datasets\water_a1_i300_380_830 ^
    --allow-underresolved ^
    --execute ^
    --resume ^
    --register-generation "bilinear_boundary_ownership_v1" ^
    --previous-generator "%BACKUP_DIR%\rainbow_generate.exe" ^
    --retry-failed ^
    --continue-on-error ^
    --recover-commits ^
    --max-consecutive-failures 400 ^
    --max-job-failures 400 ^
    --limit "%RETRY_LIMIT%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%
goto finish

:usage
echo Usage: %~nx0 backup
echo        %~nx0 retry [positive_limit]
echo Run backup BEFORE rebuilding; retry keeps successful records unchanged.
set "RESULT=2"
goto finish

:require_binaries
if not exist "%~1\rainbow_generate.exe" (
    echo Missing executable: %~1\rainbow_generate.exe
    exit /b 1
)
for %%M in (raindrop_trace.optixir patch_build.fatbin patch_query.optixir patch_optics.fatbin phase_cdf.fatbin) do (
    if not exist "%~1\modules\%%M" (
        echo Missing module: %~1\modules\%%M
        exit /b 1
    )
)
exit /b 0

:copy_checked
copy /b /y "%~1" "%~2" >nul
if errorlevel 1 exit /b 1
fc /b "%~1" "%~2" >nul
if errorlevel 1 (
    echo Backup content verification failed: %~2
    exit /b 1
)
exit /b 0

:finish
popd
endlocal & exit /b %RESULT%
