@echo off
echo Copying LawHub files to C:\LawHub_Project...
echo.

REM Use cmd.exe to run Python script
cmd /c "python copy_to_new_repo.py"

echo.
echo Copy operation completed.
pause 