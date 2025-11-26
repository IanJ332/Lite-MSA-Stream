@echo off
REM Run tests inside Docker container

echo Running tests inside Docker container...
echo.

docker exec -it lite-msa-stream python /app/testing/test_model_on_ravdess.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Test failed or container not running.
    echo Make sure the container is running with: docker-compose up -d
)
