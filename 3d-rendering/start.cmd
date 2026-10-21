@echo off
setlocal
cd /d "%~dp0"

rem Prefer a regular Node.js installation; fall back to Codex's bundled runtime.
set "XHIVE_NODE="
for /f "delims=" %%N in ('where.exe node.exe 2^>nul') do if not defined XHIVE_NODE set "XHIVE_NODE=%%N"
if not defined XHIVE_NODE if exist "%ProgramFiles%\nodejs\node.exe" set "XHIVE_NODE=%ProgramFiles%\nodejs\node.exe"
if not defined XHIVE_NODE if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "XHIVE_NODE=%LOCALAPPDATA%\Programs\nodejs\node.exe"
if not defined XHIVE_NODE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" set "XHIVE_NODE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"

if not defined XHIVE_NODE (
  echo Node.js was not found. Install Node.js 22 or newer from https://nodejs.org/
  echo Then close this window and run start.cmd again.
  pause
  exit /b 1
)

if not defined PORT set "PORT=5173"
echo XHIVE-PRO - Open http://localhost:%PORT% in your browser.
echo Keep this window open. Press Ctrl+C to stop.
echo.
"%XHIVE_NODE%" "%~dp0server.mjs"
set "XHIVE_EXIT=%ERRORLEVEL%"
if not "%XHIVE_EXIT%"=="0" (
  echo.
  echo Server exited with code %XHIVE_EXIT%. See the error above.
  echo If the port is in use, open the existing page or choose another PORT.
  pause
)
exit /b %XHIVE_EXIT%
