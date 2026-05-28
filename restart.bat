@echo off
echo =========================================
echo FaceAuth Server Restarter
echo =========================================

echo.
echo Stopping existing Spring Boot application on port 8089...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8089" ^| find "LISTENING"') do (
    echo Found process listening on port 8089: PID %%a
    taskkill /f /pid %%a
)
echo.

echo Setting JAVA_HOME to JDK 21...
set JAVA_HOME=C:\Program Files\Java\jdk-21

echo.
echo Starting Spring Boot application...
mvn spring-boot:run
