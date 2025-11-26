@echo off
REM Build script for Lite-MSA-Stream Docker image

echo ========================================
echo Building Lite-MSA-Stream Docker Image
echo ========================================

docker-compose build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build completed successfully!
    echo ========================================
    echo.
    echo To start the service, run:
    echo   docker-compose up -d
    echo.
    echo To view logs, run:
    echo   docker-compose logs -f
) else (
    echo.
    echo ========================================
    echo Build failed! Please check the errors above.
    echo ========================================
)
