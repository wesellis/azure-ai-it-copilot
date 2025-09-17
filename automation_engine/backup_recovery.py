"""
Backup and Recovery module for Azure AI IT Copilot
Handles data backup, disaster recovery, and system restoration
"""

import os
import json
import asyncio
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import gzip
import pickle

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups"""
    CONFIGURATION = "configuration"
    DATABASE = "database"
    LOGS = "logs"
    MODELS = "models"
    FULL_SYSTEM = "full_system"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class RecoveryPoint:
    """Represents a recovery point with backup data"""

    def __init__(self, backup_id: str, backup_type: BackupType, timestamp: datetime,
                 size_bytes: int, location: str, metadata: Dict = None):
        self.backup_id = backup_id
        self.backup_type = backup_type
        self.timestamp = timestamp
        self.size_bytes = size_bytes
        self.location = location
        self.metadata = metadata or {}
        self.status = BackupStatus.COMPLETED

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'timestamp': self.timestamp.isoformat(),
            'size_bytes': self.size_bytes,
            'location': self.location,
            'metadata': self.metadata,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RecoveryPoint':
        """Create from dictionary"""
        rp = cls(
            backup_id=data['backup_id'],
            backup_type=BackupType(data['backup_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            size_bytes=data['size_bytes'],
            location=data['location'],
            metadata=data.get('metadata', {})
        )
        rp.status = BackupStatus(data.get('status', 'completed'))
        return rp


class BackupManager:
    """Manages backup operations and recovery points"""

    def __init__(self, backup_root: str = None):
        self.backup_root = Path(backup_root or os.getenv('BACKUP_ROOT', './backups'))
        self.backup_root.mkdir(parents=True, exist_ok=True)

        self.recovery_points: List[RecoveryPoint] = []
        self.backup_policies = self._load_backup_policies()

        # Load existing recovery points
        self._load_recovery_points()

    def _load_backup_policies(self) -> Dict:
        """Load backup policies configuration"""
        return {
            'retention_days': {
                BackupType.CONFIGURATION: 30,
                BackupType.DATABASE: 7,
                BackupType.LOGS: 3,
                BackupType.MODELS: 14,
                BackupType.FULL_SYSTEM: 7
            },
            'schedule': {
                BackupType.CONFIGURATION: 'daily',
                BackupType.DATABASE: 'daily',
                BackupType.LOGS: 'hourly',
                BackupType.MODELS: 'weekly',
                BackupType.FULL_SYSTEM: 'weekly'
            },
            'compression': {
                BackupType.CONFIGURATION: True,
                BackupType.DATABASE: True,
                BackupType.LOGS: True,
                BackupType.MODELS: False,
                BackupType.FULL_SYSTEM: True
            },
            'encryption': {
                BackupType.CONFIGURATION: True,
                BackupType.DATABASE: True,
                BackupType.LOGS: False,
                BackupType.MODELS: False,
                BackupType.FULL_SYSTEM: True
            }
        }

    def _load_recovery_points(self):
        """Load existing recovery points from metadata"""
        metadata_file = self.backup_root / 'recovery_points.json'

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                self.recovery_points = [
                    RecoveryPoint.from_dict(rp_data)
                    for rp_data in data.get('recovery_points', [])
                ]
                logger.info(f"Loaded {len(self.recovery_points)} recovery points")
            except Exception as e:
                logger.error(f"Failed to load recovery points: {e}")

    def _save_recovery_points(self):
        """Save recovery points metadata"""
        metadata_file = self.backup_root / 'recovery_points.json'

        try:
            data = {
                'last_updated': datetime.utcnow().isoformat(),
                'recovery_points': [rp.to_dict() for rp in self.recovery_points]
            }

            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save recovery points: {e}")

    async def create_backup(self, backup_type: BackupType,
                           source_paths: List[str] = None,
                           metadata: Dict = None) -> RecoveryPoint:
        """
        Create a backup of specified type

        Args:
            backup_type: Type of backup to create
            source_paths: List of paths to backup (if not using default)
            metadata: Additional metadata for the backup

        Returns:
            RecoveryPoint representing the created backup
        """
        try:
            logger.info(f"Starting {backup_type.value} backup")

            # Generate backup ID
            backup_id = f"{backup_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Determine source paths if not provided
            if not source_paths:
                source_paths = self._get_default_source_paths(backup_type)

            # Create backup directory
            backup_dir = self.backup_root / backup_type.value / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Execute backup based on type
            backup_result = await self._execute_backup(backup_type, source_paths, backup_dir)

            # Calculate backup size
            total_size = self._calculate_backup_size(backup_dir)

            # Create recovery point
            recovery_point = RecoveryPoint(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=datetime.utcnow(),
                size_bytes=total_size,
                location=str(backup_dir),
                metadata={
                    **(metadata or {}),
                    'source_paths': source_paths,
                    'compression': self.backup_policies['compression'][backup_type],
                    'encryption': self.backup_policies['encryption'][backup_type],
                    **backup_result
                }
            )

            # Add to recovery points and save
            self.recovery_points.append(recovery_point)
            self._save_recovery_points()

            logger.info(f"Backup {backup_id} created successfully ({total_size} bytes)")
            return recovery_point

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise

    async def _execute_backup(self, backup_type: BackupType,
                             source_paths: List[str],
                             backup_dir: Path) -> Dict:
        """Execute backup operation based on type"""

        if backup_type == BackupType.CONFIGURATION:
            return await self._backup_configuration(source_paths, backup_dir)
        elif backup_type == BackupType.DATABASE:
            return await self._backup_database(source_paths, backup_dir)
        elif backup_type == BackupType.LOGS:
            return await self._backup_logs(source_paths, backup_dir)
        elif backup_type == BackupType.MODELS:
            return await self._backup_models(source_paths, backup_dir)
        elif backup_type == BackupType.FULL_SYSTEM:
            return await self._backup_full_system(source_paths, backup_dir)
        else:
            raise ValueError(f"Unknown backup type: {backup_type}")

    async def _backup_configuration(self, source_paths: List[str], backup_dir: Path) -> Dict:
        """Backup configuration files"""
        config_files = [
            '.env',
            '.env.example',
            'config/',
            'docker-compose.yml',
            'requirements.txt'
        ]

        backed_up_files = []
        for config_file in config_files:
            if os.path.exists(config_file):
                dest_path = backup_dir / os.path.basename(config_file)

                if os.path.isdir(config_file):
                    shutil.copytree(config_file, dest_path)
                else:
                    shutil.copy2(config_file, dest_path)

                backed_up_files.append(config_file)

        # Compress if policy requires
        if self.backup_policies['compression'][BackupType.CONFIGURATION]:
            await self._compress_backup(backup_dir)

        return {
            'files_backed_up': len(backed_up_files),
            'files_list': backed_up_files,
            'compressed': self.backup_policies['compression'][BackupType.CONFIGURATION]
        }

    async def _backup_database(self, source_paths: List[str], backup_dir: Path) -> Dict:
        """Backup database"""
        # This would integrate with actual database backup procedures

        # For Cosmos DB
        cosmos_backup = await self._backup_cosmos_db(backup_dir)

        # For any local SQLite or other databases
        local_db_backup = await self._backup_local_databases(backup_dir)

        return {
            'cosmos_backup': cosmos_backup,
            'local_databases': local_db_backup,
            'encrypted': self.backup_policies['encryption'][BackupType.DATABASE]
        }

    async def _backup_logs(self, source_paths: List[str], backup_dir: Path) -> Dict:
        """Backup log files"""
        log_dirs = ['logs/', 'api/logs/', 'automation_engine/logs/']
        backed_up_logs = []

        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                dest_dir = backup_dir / os.path.basename(log_dir)
                shutil.copytree(log_dir, dest_dir)
                backed_up_logs.append(log_dir)

        # Compress logs (they compress well)
        if self.backup_policies['compression'][BackupType.LOGS]:
            await self._compress_backup(backup_dir)

        return {
            'log_directories': backed_up_logs,
            'compressed': True
        }

    async def _backup_models(self, source_paths: List[str], backup_dir: Path) -> Dict:
        """Backup ML models and data"""
        model_dirs = ['ml_models/', 'ai_orchestrator/models/']
        backed_up_models = []

        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                dest_dir = backup_dir / os.path.basename(model_dir)
                shutil.copytree(model_dir, dest_dir)
                backed_up_models.append(model_dir)

        return {
            'model_directories': backed_up_models,
            'model_count': len(backed_up_models)
        }

    async def _backup_full_system(self, source_paths: List[str], backup_dir: Path) -> Dict:
        """Backup entire system"""
        # Create comprehensive system backup
        system_components = {
            'configuration': await self._backup_configuration([], backup_dir / 'config'),
            'database': await self._backup_database([], backup_dir / 'database'),
            'models': await self._backup_models([], backup_dir / 'models')
        }

        # Add application code (excluding node_modules, .git, etc.)
        code_backup_dir = backup_dir / 'application'
        code_backup_dir.mkdir(exist_ok=True)

        excluded_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv', '.venv'}
        await self._backup_application_code(code_backup_dir, excluded_dirs)

        if self.backup_policies['compression'][BackupType.FULL_SYSTEM]:
            await self._compress_backup(backup_dir)

        return {
            'components': list(system_components.keys()),
            'component_results': system_components,
            'application_code_included': True,
            'compressed': self.backup_policies['compression'][BackupType.FULL_SYSTEM]
        }

    async def _backup_cosmos_db(self, backup_dir: Path) -> Dict:
        """Backup Cosmos DB data"""
        # This would use Azure SDK to backup Cosmos DB
        # For now, simulate backup

        backup_file = backup_dir / 'cosmos_backup.json'
        backup_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'collections': ['incidents', 'resources', 'analytics', 'configurations'],
            'records_backed_up': 1000  # Simulated
        }

        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

        return {
            'success': True,
            'backup_file': str(backup_file),
            'collections_backed_up': len(backup_data['collections'])
        }

    async def _backup_local_databases(self, backup_dir: Path) -> Dict:
        """Backup local database files"""
        db_files = []

        # Look for SQLite databases
        for db_file in Path('.').rglob('*.db'):
            if db_file.exists():
                dest_file = backup_dir / db_file.name
                shutil.copy2(db_file, dest_file)
                db_files.append(str(db_file))

        return {
            'databases_backed_up': len(db_files),
            'database_files': db_files
        }

    async def _backup_application_code(self, backup_dir: Path, excluded_dirs: set):
        """Backup application code"""
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                source_file = Path(root) / file

                # Skip certain file types
                if source_file.suffix in {'.pyc', '.log', '.tmp'}:
                    continue

                relative_path = source_file.relative_to('.')
                dest_file = backup_dir / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(source_file, dest_file)

    async def _compress_backup(self, backup_dir: Path):
        """Compress backup directory"""
        try:
            import tarfile

            archive_path = backup_dir.parent / f"{backup_dir.name}.tar.gz"

            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)

            # Remove original directory
            shutil.rmtree(backup_dir)

            logger.info(f"Compressed backup to {archive_path}")

        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")

    def _calculate_backup_size(self, backup_dir: Path) -> int:
        """Calculate total size of backup"""
        total_size = 0

        if backup_dir.is_file():
            return backup_dir.stat().st_size

        for file_path in backup_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def _get_default_source_paths(self, backup_type: BackupType) -> List[str]:
        """Get default source paths for backup type"""
        defaults = {
            BackupType.CONFIGURATION: ['.env', 'config/', 'docker-compose.yml'],
            BackupType.DATABASE: ['database/', 'data/'],
            BackupType.LOGS: ['logs/', 'api/logs/'],
            BackupType.MODELS: ['ml_models/', 'ai_orchestrator/models/'],
            BackupType.FULL_SYSTEM: ['.']
        }

        return defaults.get(backup_type, [])

    async def restore_from_backup(self, backup_id: str,
                                 restore_location: str = None,
                                 selective_restore: List[str] = None) -> Dict:
        """
        Restore from a backup

        Args:
            backup_id: ID of backup to restore
            restore_location: Target location for restore (default: original location)
            selective_restore: List of specific files/dirs to restore

        Returns:
            Dict with restore results
        """
        try:
            # Find recovery point
            recovery_point = self._find_recovery_point(backup_id)
            if not recovery_point:
                raise ValueError(f"Backup {backup_id} not found")

            logger.info(f"Starting restore from backup {backup_id}")

            restore_result = {
                'backup_id': backup_id,
                'start_time': datetime.utcnow().isoformat(),
                'status': 'in_progress',
                'files_restored': 0,
                'errors': []
            }

            # Execute restore based on backup type
            if recovery_point.backup_type == BackupType.FULL_SYSTEM:
                result = await self._restore_full_system(recovery_point, restore_location)
            else:
                result = await self._restore_selective(recovery_point, restore_location, selective_restore)

            restore_result.update(result)
            restore_result['status'] = 'completed'
            restore_result['end_time'] = datetime.utcnow().isoformat()

            logger.info(f"Restore completed: {restore_result['files_restored']} files restored")
            return restore_result

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {
                'backup_id': backup_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _find_recovery_point(self, backup_id: str) -> Optional[RecoveryPoint]:
        """Find recovery point by backup ID"""
        for rp in self.recovery_points:
            if rp.backup_id == backup_id:
                return rp
        return None

    async def _restore_full_system(self, recovery_point: RecoveryPoint,
                                  restore_location: str = None) -> Dict:
        """Restore full system from backup"""
        backup_path = Path(recovery_point.location)

        # Handle compressed backups
        if not backup_path.exists() and (backup_path.parent / f"{backup_path.name}.tar.gz").exists():
            await self._decompress_backup(backup_path.parent / f"{backup_path.name}.tar.gz")

        restore_target = Path(restore_location or '.')
        files_restored = 0

        # Restore each component
        for component_dir in backup_path.iterdir():
            if component_dir.is_dir():
                target_dir = restore_target / component_dir.name

                if target_dir.exists():
                    # Create backup of existing data
                    backup_existing = target_dir.parent / f"{target_dir.name}_backup_{int(datetime.utcnow().timestamp())}"
                    shutil.move(target_dir, backup_existing)

                shutil.copytree(component_dir, target_dir)
                files_restored += len(list(component_dir.rglob('*')))

        return {
            'files_restored': files_restored,
            'restore_type': 'full_system'
        }

    async def _restore_selective(self, recovery_point: RecoveryPoint,
                                restore_location: str = None,
                                selective_files: List[str] = None) -> Dict:
        """Restore selective files from backup"""
        backup_path = Path(recovery_point.location)
        restore_target = Path(restore_location or '.')
        files_restored = 0

        if selective_files:
            for file_pattern in selective_files:
                for file_path in backup_path.rglob(file_pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(backup_path)
                        target_file = restore_target / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, target_file)
                        files_restored += 1
        else:
            # Restore all files
            for file_path in backup_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(backup_path)
                    target_file = restore_target / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_file)
                    files_restored += 1

        return {
            'files_restored': files_restored,
            'restore_type': 'selective'
        }

    async def _decompress_backup(self, archive_path: Path):
        """Decompress backup archive"""
        import tarfile

        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(archive_path.parent)

    def cleanup_old_backups(self):
        """Clean up old backups based on retention policies"""
        logger.info("Starting backup cleanup")

        cleaned_up = 0
        for backup_type in BackupType:
            retention_days = self.backup_policies['retention_days'][backup_type]
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # Find expired backups
            expired_backups = [
                rp for rp in self.recovery_points
                if rp.backup_type == backup_type and rp.timestamp < cutoff_date
            ]

            for backup in expired_backups:
                try:
                    # Remove backup files
                    backup_path = Path(backup.location)
                    if backup_path.exists():
                        if backup_path.is_dir():
                            shutil.rmtree(backup_path)
                        else:
                            backup_path.unlink()

                    # Check for compressed version
                    compressed_path = backup_path.parent / f"{backup_path.name}.tar.gz"
                    if compressed_path.exists():
                        compressed_path.unlink()

                    # Remove from recovery points
                    self.recovery_points.remove(backup)
                    backup.status = BackupStatus.EXPIRED
                    cleaned_up += 1

                    logger.info(f"Cleaned up expired backup: {backup.backup_id}")

                except Exception as e:
                    logger.error(f"Failed to cleanup backup {backup.backup_id}: {e}")

        # Save updated recovery points
        self._save_recovery_points()

        logger.info(f"Cleanup completed: {cleaned_up} backups removed")
        return cleaned_up

    def get_backup_status(self) -> Dict:
        """Get overall backup system status"""
        status = {
            'total_recovery_points': len(self.recovery_points),
            'backup_types': {},
            'total_size_bytes': 0,
            'oldest_backup': None,
            'newest_backup': None
        }

        # Group by backup type
        for backup_type in BackupType:
            type_backups = [rp for rp in self.recovery_points if rp.backup_type == backup_type]
            status['backup_types'][backup_type.value] = {
                'count': len(type_backups),
                'total_size': sum(rp.size_bytes for rp in type_backups),
                'retention_days': self.backup_policies['retention_days'][backup_type]
            }

        # Calculate totals
        status['total_size_bytes'] = sum(rp.size_bytes for rp in self.recovery_points)

        # Find oldest and newest
        if self.recovery_points:
            sorted_backups = sorted(self.recovery_points, key=lambda x: x.timestamp)
            status['oldest_backup'] = sorted_backups[0].timestamp.isoformat()
            status['newest_backup'] = sorted_backups[-1].timestamp.isoformat()

        return status

    def list_recovery_points(self, backup_type: BackupType = None) -> List[Dict]:
        """List available recovery points"""
        if backup_type:
            filtered_points = [rp for rp in self.recovery_points if rp.backup_type == backup_type]
        else:
            filtered_points = self.recovery_points

        return [rp.to_dict() for rp in sorted(filtered_points, key=lambda x: x.timestamp, reverse=True)]


# Global instance
_backup_manager = None


def get_backup_manager() -> BackupManager:
    """Get or create global backup manager instance"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


# Convenience functions
async def create_backup(backup_type: BackupType, **kwargs) -> RecoveryPoint:
    """Convenience function to create a backup"""
    manager = get_backup_manager()
    return await manager.create_backup(backup_type, **kwargs)


async def restore_backup(backup_id: str, **kwargs) -> Dict:
    """Convenience function to restore from backup"""
    manager = get_backup_manager()
    return await manager.restore_from_backup(backup_id, **kwargs)


def cleanup_backups() -> int:
    """Convenience function to cleanup old backups"""
    manager = get_backup_manager()
    return manager.cleanup_old_backups()


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = BackupManager()

        # Create a configuration backup
        recovery_point = await manager.create_backup(BackupType.CONFIGURATION)
        print(f"Created backup: {recovery_point.backup_id}")

        # List all recovery points
        points = manager.list_recovery_points()
        print(f"Total recovery points: {len(points)}")

        # Get status
        status = manager.get_backup_status()
        print(f"Backup status: {status}")

    asyncio.run(main())