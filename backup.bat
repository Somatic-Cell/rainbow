set "BACKUP=outputs\before_regular_fp64_fix"

if not exist "%BACKUP%\modules" mkdir "%BACKUP%\modules"

xcopy "build\Release\rainbow_generate.exe" "%BACKUP%\" /I /-Y
xcopy "build\Release\modules\*" "%BACKUP%\modules\" /E /I /-Y

xcopy "tools\generate_phase_dataset.py" "%BACKUP%\" /I /-Y
xcopy "configs\dataset_water_a1.json" "%BACKUP%\" /I /-Y
xcopy "datasets\water_a1_i300_380_830\batch_plan.json" "%BACKUP%\" /I /-Y