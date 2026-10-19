@echo off
rem Startet die SoundSeparator-WebApp. Doppelklick genuegt.
cd /d "%~dp0.."
call venv\Scripts\activate.bat
python webapp\app.py
pause
