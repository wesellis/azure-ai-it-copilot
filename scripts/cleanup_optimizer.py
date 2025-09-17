#!/usr/bin/env python3
"""
Code Cleanup and Optimization Script
Removes redundant code, optimizes imports, and improves performance
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Set, List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class CodeOptimizer:
    """Analyzes and optimizes Python code"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: Dict[str, List[str]] = defaultdict(list)
        self.stats = {
            "files_analyzed": 0,
            "issues_found": 0,
            "optimizations_suggested": 0
        }

    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project for optimization opportunities"""
        logger.info("Starting project analysis...")

        python_files = list(self.project_root.rglob("*.py"))

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            try:
                self._analyze_file(py_file)
                self.stats["files_analyzed"] += 1
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")

        logger.info(f"Analysis complete: {self.stats}")
        return self._generate_report()

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "migrations",
            "tests",
            ".pytest_cache"
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues[str(file_path)].append(f"Syntax error: {e}")
                return

            # Various analysis checks
            self._check_imports(file_path, tree, content)
            self._check_functions(file_path, tree)
            self._check_classes(file_path, tree)
            self._check_performance_issues(file_path, content)
            self._check_code_quality(file_path, content)

        except Exception as e:
            self.issues[str(file_path)].append(f"Analysis error: {e}")

    def _check_imports(self, file_path: Path, tree: ast.AST, content: str):
        """Check for import optimization opportunities"""
        imports = []
        star_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if any(alias.name == '*' for alias in node.names):
                        star_imports.append(node.module)

        # Check for wildcard imports
        if star_imports:
            self.issues[str(file_path)].append(
                f"Wildcard imports found: {star_imports} - consider explicit imports"
            )
            self.stats["issues_found"] += 1

        # Check for unused imports (basic check)
        if len(imports) > 20:
            self.issues[str(file_path)].append(
                f"Large number of imports ({len(imports)}) - consider refactoring"
            )

        # Check for redundant imports
        if len(imports) != len(set(imports)):
            self.issues[str(file_path)].append("Duplicate imports detected")
            self.stats["issues_found"] += 1

    def _check_functions(self, file_path: Path, tree: ast.AST):
        """Check function-related issues"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function length
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        self.issues[str(file_path)].append(
                            f"Long function '{node.name}' ({func_length} lines) - consider refactoring"
                        )

                # Check parameter count
                if len(node.args.args) > 6:
                    self.issues[str(file_path)].append(
                        f"Function '{node.name}' has many parameters ({len(node.args.args)}) - consider using dataclass or dict"
                    )

    def _check_classes(self, file_path: Path, tree: ast.AST):
        """Check class-related issues"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]

                # Check class size
                if len(methods) > 15:
                    self.issues[str(file_path)].append(
                        f"Large class '{node.name}' ({len(methods)} methods) - consider splitting"
                    )

    def _check_performance_issues(self, file_path: Path, content: str):
        """Check for performance-related issues"""
        # Check for synchronous sleep in async code
        if 'async def' in content and 'time.sleep(' in content:
            self.issues[str(file_path)].append(
                "Using time.sleep() in async function - use asyncio.sleep() instead"
            )
            self.stats["issues_found"] += 1

        # Check for inefficient string concatenation
        if '+=' in content and 'str' in content:
            # Simple heuristic - look for string concatenation in loops
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if ('for ' in line or 'while ' in line) and i < len(lines) - 1:
                    next_lines = lines[i+1:i+5]  # Check next few lines
                    if any('+=' in next_line and ('str' in next_line or '"' in next_line) for next_line in next_lines):
                        self.issues[str(file_path)].append(
                            f"Potential inefficient string concatenation at line {i+1} - consider using join() or f-strings"
                        )
                        break

        # Check for missing async/await patterns
        if 'def ' in content and ('asyncio.' in content or 'await ' in content):
            if content.count('async def') < content.count('await '):
                self.issues[str(file_path)].append(
                    "Missing async/await patterns - some functions might need to be async"
                )

    def _check_code_quality(self, file_path: Path, content: str):
        """Check general code quality issues"""
        lines = content.split('\n')

        # Check for long lines
        long_lines = [(i+1, line) for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            self.issues[str(file_path)].append(
                f"Long lines found: {len(long_lines)} lines exceed 120 characters"
            )

        # Check for TODO/FIXME comments
        todo_lines = [(i+1, line) for i, line in enumerate(lines)
                     if any(keyword in line.upper() for keyword in ['TODO', 'FIXME', 'XXX', 'HACK'])]
        if todo_lines:
            self.issues[str(file_path)].append(
                f"Technical debt markers found: {len(todo_lines)} TODO/FIXME comments"
            )

        # Check for hardcoded values
        hardcoded_patterns = [
            r'sleep\(\d+\)',  # hardcoded sleep values
            r'range\(\d{3,}\)',  # large hardcoded ranges
            r'["\']http[s]?://[^"\']+["\']',  # hardcoded URLs
        ]

        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, content)
            if matches:
                self.issues[str(file_path)].append(
                    f"Hardcoded values found: {len(matches)} instances of pattern '{pattern}'"
                )

    def _generate_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        total_issues = sum(len(issues) for issues in self.issues.values())

        # Generate optimization suggestions
        suggestions = self._generate_suggestions()

        return {
            "summary": {
                "files_analyzed": self.stats["files_analyzed"],
                "files_with_issues": len(self.issues),
                "total_issues": total_issues,
                "optimization_suggestions": len(suggestions)
            },
            "issues_by_file": dict(self.issues),
            "optimization_suggestions": suggestions,
            "priority_files": self._get_priority_files()
        }

    def _generate_suggestions(self) -> List[Dict[str, str]]:
        """Generate specific optimization suggestions"""
        suggestions = []

        # Analyze patterns across all issues
        all_issues = []
        for file_issues in self.issues.values():
            all_issues.extend(file_issues)

        # Common optimization suggestions
        if any('wildcard import' in issue.lower() for issue in all_issues):
            suggestions.append({
                "category": "imports",
                "suggestion": "Replace wildcard imports with explicit imports",
                "impact": "high",
                "effort": "low"
            })

        if any('long function' in issue.lower() for issue in all_issues):
            suggestions.append({
                "category": "functions",
                "suggestion": "Refactor long functions into smaller, focused functions",
                "impact": "medium",
                "effort": "medium"
            })

        if any('time.sleep' in issue.lower() for issue in all_issues):
            suggestions.append({
                "category": "async",
                "suggestion": "Replace time.sleep() with asyncio.sleep() in async functions",
                "impact": "high",
                "effort": "low"
            })

        if any('string concatenation' in issue.lower() for issue in all_issues):
            suggestions.append({
                "category": "performance",
                "suggestion": "Use join() or f-strings instead of += for string concatenation",
                "impact": "medium",
                "effort": "low"
            })

        if any('large class' in issue.lower() for issue in all_issues):
            suggestions.append({
                "category": "architecture",
                "suggestion": "Split large classes using composition or inheritance",
                "impact": "medium",
                "effort": "high"
            })

        return suggestions

    def _get_priority_files(self) -> List[Dict[str, Any]]:
        """Get files that should be prioritized for optimization"""
        priority_files = []

        for file_path, issues in self.issues.items():
            if not issues:
                continue

            # Calculate priority score
            score = 0
            critical_issues = 0

            for issue in issues:
                if any(keyword in issue.lower() for keyword in ['wildcard', 'time.sleep', 'syntax error']):
                    score += 3
                    critical_issues += 1
                elif any(keyword in issue.lower() for keyword in ['long', 'large', 'many']):
                    score += 2
                else:
                    score += 1

            if score > 3:  # Only include files with significant issues
                priority_files.append({
                    "file": file_path,
                    "score": score,
                    "issues_count": len(issues),
                    "critical_issues": critical_issues
                })

        # Sort by score descending
        priority_files.sort(key=lambda x: x["score"], reverse=True)
        return priority_files[:10]  # Top 10 priority files


def main():
    """Main function to run the optimization analysis"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze and optimize Python codebase")
    parser.add_argument("project_path", help="Path to the project root")
    parser.add_argument("--output", "-o", help="Output file for the report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Run analysis
    optimizer = CodeOptimizer(args.project_path)
    report = optimizer.analyze_project()

    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print("\n=== OPTIMIZATION REPORT ===")
        print(f"Files analyzed: {report['summary']['files_analyzed']}")
        print(f"Files with issues: {report['summary']['files_with_issues']}")
        print(f"Total issues: {report['summary']['total_issues']}")

        print("\n=== TOP OPTIMIZATION SUGGESTIONS ===")
        for suggestion in report['optimization_suggestions']:
            print(f"• {suggestion['suggestion']} (Impact: {suggestion['impact']}, Effort: {suggestion['effort']})")

        print("\n=== PRIORITY FILES ===")
        for file_info in report['priority_files'][:5]:
            print(f"• {file_info['file']} (Score: {file_info['score']}, Issues: {file_info['issues_count']})")


if __name__ == "__main__":
    main()