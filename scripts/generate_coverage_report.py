#!/usr/bin/env python3

"""
Azure AI IT Copilot - Coverage Report Generator
Enhanced coverage reporting with detailed analysis and visualization
"""

import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import argparse


class CoverageAnalyzer:
    """Advanced coverage analysis and reporting"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.coverage_dir = self.project_root / "htmlcov"
        self.reports_dir = self.project_root / "test-reports"
        self.coverage_data = {}

    def load_coverage_data(self) -> Dict:
        """Load coverage data from various formats"""
        coverage_data = {
            "xml": None,
            "json": None,
            "summary": {}
        }

        # Load XML coverage data
        xml_path = self.project_root / "coverage.xml"
        if xml_path.exists():
            coverage_data["xml"] = self._parse_xml_coverage(xml_path)

        # Load JSON coverage data
        json_path = self.project_root / "coverage.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                coverage_data["json"] = json.load(f)

        return coverage_data

    def _parse_xml_coverage(self, xml_path: Path) -> Dict:
        """Parse XML coverage report"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        coverage_data = {
            "timestamp": root.get("timestamp"),
            "version": root.get("version"),
            "packages": []
        }

        for package in root.findall(".//package"):
            package_data = {
                "name": package.get("name"),
                "line_rate": float(package.get("line-rate", 0)),
                "branch_rate": float(package.get("branch-rate", 0)),
                "complexity": float(package.get("complexity", 0)),
                "classes": []
            }

            for class_elem in package.findall(".//class"):
                class_data = {
                    "name": class_elem.get("name"),
                    "filename": class_elem.get("filename"),
                    "line_rate": float(class_elem.get("line-rate", 0)),
                    "branch_rate": float(class_elem.get("branch-rate", 0)),
                    "complexity": float(class_elem.get("complexity", 0))
                }
                package_data["classes"].append(class_data)

            coverage_data["packages"].append(package_data)

        return coverage_data

    def analyze_coverage_trends(self) -> Dict:
        """Analyze coverage trends and identify issues"""
        coverage_data = self.load_coverage_data()

        analysis = {
            "total_coverage": 0,
            "branch_coverage": 0,
            "file_coverage": {},
            "low_coverage_files": [],
            "uncovered_lines": [],
            "critical_gaps": [],
            "improvement_suggestions": []
        }

        if coverage_data["json"]:
            json_data = coverage_data["json"]

            # Calculate total coverage
            total_lines = json_data["totals"]["num_statements"]
            covered_lines = json_data["totals"]["covered_lines"]
            analysis["total_coverage"] = (covered_lines / total_lines * 100) if total_lines > 0 else 0

            # Calculate branch coverage
            total_branches = json_data["totals"].get("num_branches", 0)
            covered_branches = json_data["totals"].get("covered_branches", 0)
            analysis["branch_coverage"] = (covered_branches / total_branches * 100) if total_branches > 0 else 0

            # Analyze file-level coverage
            for filename, file_data in json_data["files"].items():
                if not self._should_include_file(filename):
                    continue

                file_coverage = file_data["summary"]["percent_covered"]
                analysis["file_coverage"][filename] = file_coverage

                # Identify low coverage files (below 70%)
                if file_coverage < 70:
                    analysis["low_coverage_files"].append({
                        "filename": filename,
                        "coverage": file_coverage,
                        "missing_lines": file_data["missing_lines"]
                    })

                # Identify critical gaps (below 50%)
                if file_coverage < 50:
                    analysis["critical_gaps"].append({
                        "filename": filename,
                        "coverage": file_coverage,
                        "total_lines": len(file_data["executed_lines"]) + len(file_data["missing_lines"]),
                        "missing_lines": file_data["missing_lines"]
                    })

        # Generate improvement suggestions
        analysis["improvement_suggestions"] = self._generate_suggestions(analysis)

        return analysis

    def _should_include_file(self, filename: str) -> bool:
        """Determine if file should be included in analysis"""
        exclude_patterns = [
            "test_", "_test.py", "/tests/", "__pycache__",
            ".pyc", "setup.py", "conftest.py", "/migrations/",
            "/venv/", "/env/", "/.git/", "/build/", "/dist/"
        ]

        return not any(pattern in filename for pattern in exclude_patterns)

    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []

        if analysis["total_coverage"] < 85:
            suggestions.append(
                f"Overall coverage is {analysis['total_coverage']:.1f}%. "
                "Consider adding tests to reach the 85% target."
            )

        if analysis["branch_coverage"] < 80:
            suggestions.append(
                f"Branch coverage is {analysis['branch_coverage']:.1f}%. "
                "Add tests for conditional logic and error handling paths."
            )

        if len(analysis["critical_gaps"]) > 0:
            suggestions.append(
                f"Found {len(analysis['critical_gaps'])} files with critical coverage gaps (<50%). "
                "Priority should be given to testing these files."
            )

        if len(analysis["low_coverage_files"]) > 5:
            suggestions.append(
                f"Found {len(analysis['low_coverage_files'])} files with low coverage (<70%). "
                "Consider implementing focused testing sessions for these files."
            )

        return suggestions

    def generate_detailed_report(self) -> str:
        """Generate comprehensive coverage report"""
        analysis = self.analyze_coverage_trends()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
# Azure AI IT Copilot - Coverage Analysis Report
Generated: {timestamp}

## Executive Summary

### Overall Metrics
- **Total Coverage**: {analysis['total_coverage']:.2f}%
- **Branch Coverage**: {analysis['branch_coverage']:.2f}%
- **Files Analyzed**: {len(analysis['file_coverage'])}
- **Low Coverage Files**: {len(analysis['low_coverage_files'])}
- **Critical Gaps**: {len(analysis['critical_gaps'])}

### Coverage Status
"""

        if analysis["total_coverage"] >= 85:
            report += "✅ **PASSED** - Coverage meets minimum threshold\n"
        else:
            report += "❌ **FAILED** - Coverage below minimum threshold (85%)\n"

        if analysis["branch_coverage"] >= 80:
            report += "✅ **PASSED** - Branch coverage meets recommended threshold\n"
        else:
            report += "⚠️ **WARNING** - Branch coverage below recommended threshold (80%)\n"

        # Low coverage files section
        if analysis["low_coverage_files"]:
            report += "\n## Low Coverage Files (< 70%)\n\n"
            report += "| File | Coverage | Priority |\n"
            report += "|------|----------|----------|\n"

            # Sort by coverage (lowest first)
            sorted_files = sorted(analysis["low_coverage_files"], key=lambda x: x["coverage"])

            for file_info in sorted_files[:10]:  # Show top 10
                filename = file_info["filename"]
                coverage = file_info["coverage"]

                if coverage < 30:
                    priority = "🔴 Critical"
                elif coverage < 50:
                    priority = "🟡 High"
                else:
                    priority = "🟢 Medium"

                report += f"| `{filename}` | {coverage:.1f}% | {priority} |\n"

        # Critical gaps section
        if analysis["critical_gaps"]:
            report += "\n## Critical Coverage Gaps (< 50%)\n\n"

            for gap in analysis["critical_gaps"]:
                report += f"### {gap['filename']}\n"
                report += f"- **Coverage**: {gap['coverage']:.1f}%\n"
                report += f"- **Total Lines**: {gap['total_lines']}\n"
                report += f"- **Missing Lines**: {len(gap['missing_lines'])}\n"

                if len(gap['missing_lines']) <= 20:
                    report += f"- **Uncovered Lines**: {', '.join(map(str, gap['missing_lines']))}\n"
                else:
                    report += f"- **Uncovered Lines**: {', '.join(map(str, gap['missing_lines'][:10]))}... (+{len(gap['missing_lines'])-10} more)\n"

                report += "\n"

        # Top performing files
        if analysis["file_coverage"]:
            top_files = sorted(
                [(f, c) for f, c in analysis["file_coverage"].items() if c > 90],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            if top_files:
                report += "\n## Top Performing Files (> 90%)\n\n"
                report += "| File | Coverage |\n"
                report += "|------|----------|\n"

                for filename, coverage in top_files:
                    report += f"| `{filename}` | {coverage:.1f}% |\n"

        # Improvement suggestions
        if analysis["improvement_suggestions"]:
            report += "\n## Improvement Suggestions\n\n"
            for i, suggestion in enumerate(analysis["improvement_suggestions"], 1):
                report += f"{i}. {suggestion}\n"

        # Technical recommendations
        report += "\n## Technical Recommendations\n\n"
        report += """
### Immediate Actions
1. **Focus on Critical Gaps**: Prioritize files with <50% coverage
2. **Add Integration Tests**: Improve branch coverage through integration testing
3. **Mock External Dependencies**: Use mocking to test error conditions
4. **Test Edge Cases**: Add tests for boundary conditions and error paths

### Testing Strategy
1. **Unit Tests**: Aim for 90%+ coverage on core business logic
2. **Integration Tests**: Cover service interactions and workflows
3. **Error Handling**: Test all exception paths and error conditions
4. **Performance Tests**: Include performance-critical code paths

### Monitoring
1. **Coverage Trends**: Track coverage changes over time
2. **Quality Gates**: Enforce coverage thresholds in CI/CD
3. **Regular Reviews**: Conduct weekly coverage analysis
4. **Team Metrics**: Share coverage goals across development team
"""

        return report

    def save_report(self, report_content: str, filename: str = "coverage-analysis-report.md"):
        """Save the detailed report to file"""
        self.reports_dir.mkdir(exist_ok=True)
        report_path = self.reports_dir / filename

        with open(report_path, 'w') as f:
            f.write(report_content)

        return report_path

    def generate_coverage_badge(self) -> str:
        """Generate coverage badge data"""
        analysis = self.analyze_coverage_trends()
        coverage = analysis["total_coverage"]

        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellowgreen"
        elif coverage >= 60:
            color = "yellow"
        elif coverage >= 50:
            color = "orange"
        else:
            color = "red"

        badge_url = f"https://img.shields.io/badge/coverage-{coverage:.0f}%25-{color}"

        return {
            "coverage": coverage,
            "color": color,
            "url": badge_url,
            "markdown": f"![Coverage]({badge_url})"
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate comprehensive coverage analysis report")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="coverage-analysis-report.md", help="Output report filename")
    parser.add_argument("--badge", action="store_true", help="Generate coverage badge")
    parser.add_argument("--json", action="store_true", help="Output analysis as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CoverageAnalyzer(args.project_root)

    if args.verbose:
        print("Loading coverage data...")

    # Generate analysis
    analysis = analyzer.analyze_coverage_trends()

    if args.json:
        # Output as JSON
        print(json.dumps(analysis, indent=2))
        return

    if args.badge:
        # Generate badge information
        badge_info = analyzer.generate_coverage_badge()
        print(f"Coverage: {badge_info['coverage']:.1f}%")
        print(f"Badge URL: {badge_info['url']}")
        print(f"Markdown: {badge_info['markdown']}")
        return

    # Generate detailed report
    if args.verbose:
        print("Generating detailed report...")

    report_content = analyzer.generate_detailed_report()
    report_path = analyzer.save_report(report_content, args.output)

    print(f"Coverage analysis report generated: {report_path}")

    # Display summary
    print(f"\nSummary:")
    print(f"  Total Coverage: {analysis['total_coverage']:.2f}%")
    print(f"  Branch Coverage: {analysis['branch_coverage']:.2f}%")
    print(f"  Low Coverage Files: {len(analysis['low_coverage_files'])}")
    print(f"  Critical Gaps: {len(analysis['critical_gaps'])}")

    if analysis['total_coverage'] < 85:
        print(f"\n⚠️  Coverage below target threshold (85%)")
        exit(1)
    else:
        print(f"\n✅ Coverage meets target threshold")


if __name__ == "__main__":
    main()