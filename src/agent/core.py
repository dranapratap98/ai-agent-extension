from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import ast
import asyncio

class OperationType(str, Enum):
    SUGGEST = "suggest"
    EDIT = "edit"
    TEST = "test"
    INSTALL = "install"
    EXECUTE = "execute"

class CodeOperation(BaseModel):
    """Represents a code operation request."""
    operation_type: OperationType
    content: str
    file_path: Optional[str] = None
    language: str
    cursor_position: Optional[int] = None
    selected_text: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class OperationResult(BaseModel):
    """Result of a code operation."""
    success: bool
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class CodeAnalyzer:
    """Analyzes code and provides context for AI operations."""
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """Extract import statements from Python code."""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(name.name for name in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend(f"{module}.{name.name}" for name in node.names)
            return imports
        except SyntaxError:
            return []

    @staticmethod
    def get_context_window(code: str, position: int, window_size: int = 100) -> str:
        """Get code context around a specific position."""
        start = max(0, position - window_size)
        end = min(len(code), position + window_size)
        return code[start:end]

class AIAgent:
    """Core AI Agent class that coordinates operations and providers."""
    
    def __init__(self, provider, executor):
        self.provider = provider
        self.executor = executor
        self.analyzer = CodeAnalyzer()
        self._operation_handlers = {
            OperationType.SUGGEST: self._handle_suggest,
            OperationType.EDIT: self._handle_edit,
            OperationType.TEST: self._handle_test,
            OperationType.INSTALL: self._handle_install,
            OperationType.EXECUTE: self._handle_execute,
        }

    async def execute_operation(self, operation: CodeOperation) -> OperationResult:
        """Process a code operation request."""
        try:
            handler = self._operation_handlers.get(operation.operation_type)
            if not handler:
                return OperationResult(success=False, content="", errors=[f"Unsupported operation type: {operation.operation_type}"])
            return await handler(operation)
        except Exception as e:
            return OperationResult(success=False, content="", errors=[f"Operation failed: {str(e)}"])

    async def _handle_suggest(self, operation: CodeOperation) -> OperationResult:
        """Handle code suggestion requests."""
        imports = self.analyzer.extract_imports(operation.content)
        prompt = f"""Given the following code context in {operation.language}:

{operation.content}

Current imports: {', '.join(imports)}
Cursor position: {operation.cursor_position}

Please suggest completions or improvements for this code."""
        response = await self.provider.generate_response(prompt)
        return OperationResult(success=True, content=response, metadata={"imports": imports})

    async def _handle_edit(self, operation: CodeOperation) -> OperationResult:
        """Handle code editing requests."""
        window = self.analyzer.get_context_window(operation.content, operation.cursor_position) if operation.cursor_position else None
        prompt = f"""Edit the following code in {operation.language}:

{operation.content}

{'Context around cursor:' if window else ''}
{window if window else ''}

Please provide the edited code with improvements or fixes."""
        response = await self.provider.generate_response(prompt)
        return OperationResult(success=True, content=response, metadata={"modified_at": operation.cursor_position})

    async def _handle_test(self, operation: CodeOperation) -> OperationResult:
        """Handle test generation requests."""
        prompt = f"""Generate tests for the following {operation.language} code:

{operation.content}

Please provide comprehensive test cases covering the main functionality."""
        response = await self.provider.generate_response(prompt)
        return OperationResult(success=True, content=response, metadata={"test_framework": "pytest"})

    async def _handle_install(self, operation: CodeOperation) -> OperationResult:
        """Handle package installation requests."""
        result = await self.executor.install_dependencies(operation.parameters.get("packages", []))
        return OperationResult(success=result.get("success", False), content=result.get("output", ""), metadata={"installed_packages": result.get("installed", [])})

    async def _handle_execute(self, operation: CodeOperation) -> OperationResult:
        """Handle code execution requests."""
        result = await self.executor.execute_code(operation.content, operation.language)
        return OperationResult(success=result.get("success", False), content=result.get("output", ""), metadata={"execution_time": result.get("time", 0), "memory_usage": result.get("memory", 0)})
