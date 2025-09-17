"""
Configuration validation and management for Azure AI IT Copilot
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a configuration validation"""
    level: ValidationLevel
    key: str
    message: str
    suggestion: Optional[str] = None


class ConfigValidator:
    """Validates environment configuration"""

    def __init__(self):
        self.required_vars = {
            # Azure Authentication (Critical)
            'AZURE_SUBSCRIPTION_ID': {
                'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'description': 'Azure subscription ID (UUID format)',
                'critical': True
            },
            'AZURE_TENANT_ID': {
                'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'description': 'Azure tenant ID (UUID format)',
                'critical': True
            },

            # Azure OpenAI (Critical for AI functionality)
            'AZURE_OPENAI_ENDPOINT': {
                'pattern': r'^https://[\w\-]+\.openai\.azure\.com/?$',
                'description': 'Azure OpenAI endpoint URL',
                'critical': True
            },
            'AZURE_OPENAI_KEY': {
                'pattern': r'^[a-zA-Z0-9]{32,}$',
                'description': 'Azure OpenAI API key',
                'critical': True,
                'sensitive': True
            },

            # Security (Critical)
            'JWT_SECRET_KEY': {
                'min_length': 32,
                'description': 'JWT secret key for authentication',
                'critical': True,
                'sensitive': True
            },

            # Application (Required)
            'API_HOST': {
                'default': '0.0.0.0',
                'description': 'API host address'
            },
            'API_PORT': {
                'pattern': r'^[0-9]{1,5}$',
                'default': '8000',
                'description': 'API port number'
            }
        }

        self.optional_vars = {
            # Azure Services (Optional but recommended)
            'AZURE_CLIENT_ID': {
                'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'description': 'Azure service principal client ID'
            },
            'AZURE_CLIENT_SECRET': {
                'min_length': 16,
                'description': 'Azure service principal secret',
                'sensitive': True
            },

            # Database (Optional)
            'COSMOS_DB_ENDPOINT': {
                'pattern': r'^https://[\w\-]+\.documents\.azure\.com:443/?$',
                'description': 'Cosmos DB endpoint URL'
            },
            'REDIS_HOST': {
                'default': 'localhost',
                'description': 'Redis host address'
            },
            'REDIS_PORT': {
                'pattern': r'^[0-9]{1,5}$',
                'default': '6379',
                'description': 'Redis port number'
            },

            # Feature Flags
            'ENABLE_AUTO_REMEDIATION': {
                'pattern': r'^(true|false)$',
                'default': 'true',
                'description': 'Enable automatic incident remediation'
            },
            'ENABLE_PREDICTIVE_ANALYTICS': {
                'pattern': r'^(true|false)$',
                'default': 'true',
                'description': 'Enable predictive analytics features'
            }
        }

    def validate_environment(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate current environment configuration

        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = []
        is_valid = True

        # Check required variables
        for var_name, config in self.required_vars.items():
            result = self._validate_variable(var_name, config, required=True)
            results.append(result)

            if result.level == ValidationLevel.ERROR:
                is_valid = False

        # Check optional variables if present
        for var_name, config in self.optional_vars.items():
            if os.getenv(var_name):
                result = self._validate_variable(var_name, config, required=False)
                results.append(result)

        # Additional validation checks
        results.extend(self._validate_security_settings())
        results.extend(self._validate_azure_connectivity())
        results.extend(self._validate_feature_flags())

        return is_valid, results

    def _validate_variable(self, var_name: str, config: Dict, required: bool = True) -> ValidationResult:
        """Validate a single environment variable"""
        value = os.getenv(var_name)

        if not value:
            if required:
                return ValidationResult(
                    level=ValidationLevel.ERROR,
                    key=var_name,
                    message=f"Required variable {var_name} is not set",
                    suggestion=f"Set {var_name}: {config['description']}"
                )
            else:
                default = config.get('default')
                if default:
                    return ValidationResult(
                        level=ValidationLevel.INFO,
                        key=var_name,
                        message=f"Optional variable {var_name} not set, using default: {default}"
                    )
                return ValidationResult(
                    level=ValidationLevel.INFO,
                    key=var_name,
                    message=f"Optional variable {var_name} not set"
                )

        # Check pattern if specified
        if 'pattern' in config:
            if not re.match(config['pattern'], value):
                return ValidationResult(
                    level=ValidationLevel.ERROR,
                    key=var_name,
                    message=f"{var_name} does not match required pattern",
                    suggestion=f"Expected format: {config['description']}"
                )

        # Check minimum length if specified
        if 'min_length' in config:
            if len(value) < config['min_length']:
                return ValidationResult(
                    level=ValidationLevel.ERROR,
                    key=var_name,
                    message=f"{var_name} is too short (minimum {config['min_length']} characters)"
                )

        # Check for placeholder values
        placeholder_patterns = [
            r'your-.*-here',
            r'change-.*-production',
            r'development-.*',
            r'placeholder'
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                level = ValidationLevel.ERROR if config.get('critical') else ValidationLevel.WARNING
                return ValidationResult(
                    level=level,
                    key=var_name,
                    message=f"{var_name} appears to contain a placeholder value",
                    suggestion="Replace with actual production value"
                )

        return ValidationResult(
            level=ValidationLevel.INFO,
            key=var_name,
            message=f"{var_name} is valid"
        )

    def _validate_security_settings(self) -> List[ValidationResult]:
        """Validate security-related settings"""
        results = []

        # Check JWT secret strength
        jwt_secret = os.getenv('JWT_SECRET_KEY')
        if jwt_secret:
            if len(jwt_secret) < 32:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='JWT_SECRET_KEY',
                    message="JWT secret key is too weak",
                    suggestion="Use a secret key with at least 32 characters"
                ))
            elif 'development' in jwt_secret.lower():
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key='JWT_SECRET_KEY',
                    message="JWT secret appears to be a development key",
                    suggestion="Use a strong, unique secret for production"
                ))

        # Check for development passwords
        admin_password = os.getenv('ADMIN_PASSWORD')
        if admin_password and admin_password in ['admin', 'admin123', 'password', '123456']:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='ADMIN_PASSWORD',
                message="Admin password is insecure",
                suggestion="Use ADMIN_PASSWORD_HASH with a bcrypt hashed password"
            ))

        # Check environment setting
        environment = os.getenv('ENVIRONMENT', 'development')
        if environment == 'production':
            # Additional production checks
            if os.getenv('DEBUG', 'false').lower() == 'true':
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key='DEBUG',
                    message="Debug mode is enabled in production",
                    suggestion="Set DEBUG=false for production"
                ))

        return results

    def _validate_azure_connectivity(self) -> List[ValidationResult]:
        """Validate Azure service connectivity (basic checks)"""
        results = []

        # Check if required Azure variables are present for authentication
        subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        if subscription_id and tenant_id:
            # Check if we have either service principal or managed identity setup
            client_id = os.getenv('AZURE_CLIENT_ID')
            client_secret = os.getenv('AZURE_CLIENT_SECRET')

            if not (client_id and client_secret):
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key='AZURE_AUTH',
                    message="No service principal credentials found",
                    suggestion="Set AZURE_CLIENT_ID and AZURE_CLIENT_SECRET, or ensure managed identity is configured"
                ))

        return results

    def _validate_feature_flags(self) -> List[ValidationResult]:
        """Validate feature flag configurations"""
        results = []

        feature_flags = [
            'ENABLE_AUTO_REMEDIATION',
            'ENABLE_PREDICTIVE_ANALYTICS',
            'ENABLE_COST_OPTIMIZATION',
            'ENABLE_COMPLIANCE_CHECKS'
        ]

        for flag in feature_flags:
            value = os.getenv(flag)
            if value and value.lower() not in ['true', 'false']:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key=flag,
                    message=f"Feature flag {flag} has invalid value: {value}",
                    suggestion="Use 'true' or 'false'"
                ))

        return results

    def generate_env_template(self) -> str:
        """Generate a .env template with all required and optional variables"""
        template = []
        template.append("# Azure AI IT Copilot Configuration")
        template.append("# Generated configuration template")
        template.append("")

        # Required variables
        template.append("# ================================")
        template.append("# REQUIRED CONFIGURATION")
        template.append("# ================================")
        template.append("")

        for var_name, config in self.required_vars.items():
            template.append(f"# {config['description']}")
            if config.get('sensitive'):
                template.append(f"{var_name}=your-secret-here-replace-this")
            else:
                default = config.get('default', f'your-{var_name.lower().replace("_", "-")}-here')
                template.append(f"{var_name}={default}")
            template.append("")

        # Optional variables
        template.append("# ================================")
        template.append("# OPTIONAL CONFIGURATION")
        template.append("# ================================")
        template.append("")

        for var_name, config in self.optional_vars.items():
            template.append(f"# {config['description']}")
            default = config.get('default', '')
            if config.get('sensitive'):
                template.append(f"# {var_name}=your-secret-here")
            else:
                template.append(f"# {var_name}={default}")
            template.append("")

        return "\n".join(template)

    def print_validation_report(self, results: List[ValidationResult]) -> None:
        """Print a formatted validation report"""
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        info = [r for r in results if r.level == ValidationLevel.INFO]

        print("\n" + "="*60)
        print("AZURE AI IT COPILOT - CONFIGURATION VALIDATION REPORT")
        print("="*60)

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for result in errors:
                print(f"  • {result.key}: {result.message}")
                if result.suggestion:
                    print(f"    ↳ {result.suggestion}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for result in warnings:
                print(f"  • {result.key}: {result.message}")
                if result.suggestion:
                    print(f"    ↳ {result.suggestion}")

        if info:
            print(f"\n✅ VALID ({len(info)}):")
            for result in info:
                if not result.message.endswith("is valid"):
                    print(f"  • {result.key}: {result.message}")

        print(f"\n📊 SUMMARY:")
        print(f"  • Total checks: {len(results)}")
        print(f"  • Errors: {len(errors)}")
        print(f"  • Warnings: {len(warnings)}")
        print(f"  • Valid: {len(info)}")

        if errors:
            print(f"\n🚨 Configuration is INVALID - {len(errors)} errors must be fixed")
        elif warnings:
            print(f"\n⚠️  Configuration is USABLE but has {len(warnings)} warnings")
        else:
            print(f"\n✅ Configuration is VALID and ready for production")

        print("="*60)


def validate_config() -> bool:
    """
    Main configuration validation function

    Returns:
        True if configuration is valid, False otherwise
    """
    validator = ConfigValidator()
    is_valid, results = validator.validate_environment()
    validator.print_validation_report(results)
    return is_valid


if __name__ == "__main__":
    # Run validation when script is executed directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--generate-template':
        validator = ConfigValidator()
        template = validator.generate_env_template()
        print(template)
    else:
        valid = validate_config()
        sys.exit(0 if valid else 1)