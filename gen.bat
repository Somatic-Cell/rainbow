generate_dataset.bat ^
    --config configs\dataset_water_a1.json ^
    --inclination-count 300 ^
    --wavelength-range 380 830 5 ^
    --out-root datasets\water_a1_i300_380_830 ^
    --allow-underresolved ^
    --execute ^
    --resume ^
    --continue-on-error ^
    --recover-commits ^
    --max-consecutive-failures 200 ^
    --max-job-failures 1000

echo Exit code: %ERRORLEVEL%