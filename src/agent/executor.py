import asyncio
import subprocess
import sys
import os
import tempfile
import resource
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionError(Exception):
    """Custom exception for execution errors."""
    pass

class ResourceLimits:
    """Resource limits for code execution."""
    MAX_MEMORY = 512 * 1024 * 1024  # 512MB
    MAX_TIME = 30  # seconds
    MAX_PROCESSES = 5

class ExecutionResult:
    """Stores the result of code execution."""
    def __init__(self, success: bool, output: str, error: Optional[str] = None, execution_time: float = 0, memory_usage: int = 0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.memory_usage = memory_usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "time": self.execution_time,
            "memory": self.memory_usage
        }

class CodeExecutor:
    """Handles safe code execution in isolated environments."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="ai_agent_")
        self.supported_languages = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "shell": self._execute_shell
        }
        
    def _apply_limits(self):
        """Apply resource limits to the process."""
        resource.setrlimit(resource.RLIMIT_AS, (ResourceLimits.MAX_MEMORY, ResourceLimits.MAX_MEMORY))
        resource.setrlimit(resource.RLIMIT_CPU, (ResourceLimits.MAX_TIME, ResourceLimits.MAX_TIME))
        resource.setrlimit(resource.RLIMIT_NPROC, (ResourceLimits.MAX_PROCESSES, ResourceLimits.MAX_PROCESSES))

    async def execute_code(self, code: str, language: str) -> ExecutionResult:
        """Execute code in the specified language."""
        executor = self.supported_languages.get(language.lower())
        if not executor:
            return ExecutionResult(success=False, output="", error=f"Unsupported language: {language}")
        
        try:
            return await executor(code)
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return ExecutionResult(success=False, output="", error=str(e))

    async def _execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code in a safe environment."""
        temp_file = Path(self.workspace_dir) / "script.py"
        
        try:
            temp_file.write_text(code)
            
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(temp_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._apply_limits if sys.platform != "win32" else None
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=ResourceLimits.MAX_TIME)
            return ExecutionResult(success=process.returncode == 0, output=stdout.decode(), error=stderr.decode() if stderr else None)
        except asyncio.TimeoutError:
            process.terminate()
            return ExecutionResult(success=False, output="", error="Execution timeout")
        finally:
            if temp_file.exists():
                temp_file.unlink()

    async def _execute_javascript(self, code: str) -> ExecutionResult:
        """Execute JavaScript code using Node.js."""
        temp_file = Path(self.workspace_dir) / "script.js"
        
        try:
            temp_file.write_text(code)
            
            process = await asyncio.create_subprocess_exec(
                "node",
                str(temp_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._apply_limits if sys.platform != "win32" else None
            )
            
            stdout, stderr = await process.communicate()
            return ExecutionResult(success=process.returncode == 0, output=stdout.decode(), error=stderr.decode() if stderr else None)
        except FileNotFoundError:
            return ExecutionResult(success=False, output="", error="Node.js is not installed")
        finally:
            if temp_file.exists():
                temp_file.unlink()

    async def _execute_shell(self, code: str) -> ExecutionResult:
        """Execute shell commands with safety checks."""
        allowed_commands = {"echo", "ls", "pwd", "whoami", "date"}
        if any(cmd not in allowed_commands for cmd in code.split()):
            return ExecutionResult(success=False, output="", error="Unsafe command detected")
        
        process = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=self._apply_limits if sys.platform != "win32" else None
        )
        
        stdout, stderr = await process.communicate()
        return ExecutionResult(success=process.returncode == 0, output=stdout.decode(), error=stderr.decode() if stderr else None)

class DependencyManager:
    """Manages package installations and dependencies."""
    
    def __init__(self):
        self.package_managers = {
            "python": ("pip", "install"),
            "javascript": ("npm", "install"),
        }
    
    async def install_dependencies(self, packages: List[str], language: str = "python") -> Dict[str, Any]:
        """Install packages using the appropriate package manager."""
        if not packages:
            return {"success": True, "output": "No packages to install"}
        
        if language not in self.package_managers:
            return {"success": False, "output": f"Unsupported language: {language}"}
        
        package_manager, install_cmd = self.package_managers[language]
        
        try:
            process = await asyncio.create_subprocess_exec(
                package_manager,
                install_cmd,
                *packages,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            return {
                "success": process.returncode == 0,
                "output": stdout.decode(),
                "error": stderr.decode() if stderr else None,
                "installed": packages if process.returncode == 0 else []
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "installed": []}

class Executor:
    """Main executor class that coordinates code execution and dependency management."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.code_executor = CodeExecutor(workspace_dir)
        self.dependency_manager = DependencyManager()
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="ai_agent_")
        
    async def execute_code(self, code: str, language: str) -> Dict[str, Any]:
        """Execute code and return results."""
        result = await self.code_executor.execute_code(code, language)
        return result.to_dict()
    
    async def install_dependencies(self, packages: List[str], language: str = "python") -> Dict[str, Any]:
        """Install dependencies and return results."""
        return await self.dependency_manager.install_dependencies(packages, language)
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
