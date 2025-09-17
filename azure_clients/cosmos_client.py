"""
Azure Cosmos DB Client
Handles Cosmos DB operations with proper error handling and async support
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from azure.cosmos import CosmosClient
from azure.cosmos import PartitionKey, exceptions
from azure.core.exceptions import ResourceNotFoundError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CosmosDBClient:
    """Azure Cosmos DB Client with async support"""

    def __init__(self):
        """Initialize Cosmos DB client"""
        self.settings = get_settings()
        self.client = None
        self.database = None
        self._initialized = False

        # Container definitions
        self.containers = {
            'commands': {
                'id': 'commands',
                'partition_key': PartitionKey(path="/user_id"),
                'default_ttl': 2592000  # 30 days
            },
            'incidents': {
                'id': 'incidents',
                'partition_key': PartitionKey(path="/severity"),
                'default_ttl': 7776000  # 90 days
            },
            'cost_analysis': {
                'id': 'cost_analysis',
                'partition_key': PartitionKey(path="/scope"),
                'default_ttl': 5184000  # 60 days
            },
            'compliance_reports': {
                'id': 'compliance_reports',
                'partition_key': PartitionKey(path="/framework"),
                'default_ttl': 15552000  # 180 days
            },
            'user_sessions': {
                'id': 'user_sessions',
                'partition_key': PartitionKey(path="/user_id"),
                'default_ttl': 86400  # 1 day
            },
            'analytics': {
                'id': 'analytics',
                'partition_key': PartitionKey(path="/metric_type"),
                'default_ttl': 7776000  # 90 days
            }
        }

    async def initialize(self):
        """Initialize Cosmos DB client and database"""
        if self._initialized:
            return

        try:
            # Initialize client
            self.client = CosmosClient(
                self.settings.cosmos_db_endpoint,
                self.settings.cosmos_db_key
            )

            # Create or get database
            self.database = await self.client.create_database_if_not_exists(
                id=self.settings.cosmos_db_name
            )

            # Create containers
            await self._create_containers()

            self._initialized = True
            logger.info("Cosmos DB client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {str(e)}")
            raise

    async def close(self):
        """Close Cosmos DB client"""
        if self.client:
            await self.client.close()
        self._initialized = False
        logger.info("Cosmos DB client closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _create_containers(self):
        """Create all required containers"""
        for container_name, config in self.containers.items():
            try:
                await self.database.create_container_if_not_exists(
                    id=config['id'],
                    partition_key=config['partition_key'],
                    default_ttl=config.get('default_ttl')
                )
                logger.debug(f"Container '{container_name}' ready")
            except Exception as e:
                logger.error(f"Failed to create container '{container_name}': {str(e)}")
                raise

    async def store_command_result(self, command_data: Dict[str, Any]) -> str:
        """Store command execution result"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('commands')

            # Add metadata
            command_data.update({
                'id': command_data.get('request_id'),
                'created_at': datetime.utcnow().isoformat(),
                'ttl': 2592000  # 30 days
            })

            result = await container.create_item(command_data)
            logger.info(f"Stored command result: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing command result: {str(e)}")
            raise

    async def get_command_result(self, command_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get command execution result"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('commands')
            result = await container.read_item(command_id, partition_key=user_id)
            return result

        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Command result not found: {command_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting command result: {str(e)}")
            raise

    async def store_incident(self, incident_data: Dict[str, Any]) -> str:
        """Store incident information"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('incidents')

            # Add metadata
            incident_data.update({
                'id': incident_data.get('incident_id', f"incident_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                'created_at': datetime.utcnow().isoformat(),
                'status': incident_data.get('status', 'open'),
                'ttl': 7776000  # 90 days
            })

            result = await container.create_item(incident_data)
            logger.info(f"Stored incident: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing incident: {str(e)}")
            raise

    async def update_incident_status(self, incident_id: str, status: str, severity: str) -> bool:
        """Update incident status"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('incidents')

            # Read current incident
            incident = await container.read_item(incident_id, partition_key=severity)

            # Update status
            incident['status'] = status
            incident['updated_at'] = datetime.utcnow().isoformat()

            if status == 'resolved':
                incident['resolved_at'] = datetime.utcnow().isoformat()

            # Update item
            await container.replace_item(incident_id, incident)
            logger.info(f"Updated incident {incident_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating incident status: {str(e)}")
            raise

    async def get_active_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active incidents"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('incidents')

            query = "SELECT * FROM c WHERE c.status != 'resolved' ORDER BY c.created_at DESC"
            items = []

            async for item in container.query_items(query, max_item_count=limit):
                items.append(item)

            logger.info(f"Retrieved {len(items)} active incidents")
            return items

        except Exception as e:
            logger.error(f"Error getting active incidents: {str(e)}")
            raise

    async def store_cost_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Store cost analysis results"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('cost_analysis')

            # Add metadata
            analysis_data.update({
                'id': f"cost_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'created_at': datetime.utcnow().isoformat(),
                'ttl': 5184000  # 60 days
            })

            result = await container.create_item(analysis_data)
            logger.info(f"Stored cost analysis: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing cost analysis: {str(e)}")
            raise

    async def get_cost_history(self, scope: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get cost analysis history"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('cost_analysis')

            query = f"""
                SELECT * FROM c
                WHERE c.scope = @scope
                AND c.created_at >= DateTimeAdd('day', -{days}, GetCurrentDateTime())
                ORDER BY c.created_at DESC
            """

            items = []
            async for item in container.query_items(
                query,
                parameters=[{"name": "@scope", "value": scope}],
                max_item_count=100
            ):
                items.append(item)

            logger.info(f"Retrieved {len(items)} cost history records")
            return items

        except Exception as e:
            logger.error(f"Error getting cost history: {str(e)}")
            raise

    async def store_compliance_report(self, report_data: Dict[str, Any]) -> str:
        """Store compliance report"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('compliance_reports')

            # Add metadata
            report_data.update({
                'id': f"compliance_{report_data.get('framework')}_{datetime.utcnow().strftime('%Y%m%d')}",
                'created_at': datetime.utcnow().isoformat(),
                'ttl': 15552000  # 180 days
            })

            result = await container.create_item(report_data)
            logger.info(f"Stored compliance report: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing compliance report: {str(e)}")
            raise

    async def get_latest_compliance_report(self, framework: str) -> Optional[Dict[str, Any]]:
        """Get latest compliance report for framework"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('compliance_reports')

            query = """
                SELECT TOP 1 * FROM c
                WHERE c.framework = @framework
                ORDER BY c.created_at DESC
            """

            items = []
            async for item in container.query_items(
                query,
                parameters=[{"name": "@framework", "value": framework}]
            ):
                items.append(item)

            return items[0] if items else None

        except Exception as e:
            logger.error(f"Error getting compliance report: {str(e)}")
            raise

    async def store_user_session(self, session_data: Dict[str, Any]) -> str:
        """Store user session data"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('user_sessions')

            # Add metadata
            session_data.update({
                'id': session_data.get('session_id'),
                'created_at': datetime.utcnow().isoformat(),
                'ttl': 86400  # 1 day
            })

            result = await container.create_item(session_data)
            logger.info(f"Stored user session: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing user session: {str(e)}")
            raise

    async def get_user_command_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get command history for user"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('commands')

            query = """
                SELECT * FROM c
                WHERE c.user_id = @user_id
                ORDER BY c.created_at DESC
            """

            items = []
            async for item in container.query_items(
                query,
                parameters=[{"name": "@user_id", "value": user_id}],
                max_item_count=limit
            ):
                items.append(item)

            logger.info(f"Retrieved {len(items)} commands for user {user_id}")
            return items

        except Exception as e:
            logger.error(f"Error getting user command history: {str(e)}")
            raise

    async def store_analytics_data(self, analytics_data: Dict[str, Any]) -> str:
        """Store analytics metrics"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('analytics')

            # Add metadata
            analytics_data.update({
                'id': f"analytics_{analytics_data.get('metric_type')}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                'created_at': datetime.utcnow().isoformat(),
                'ttl': 7776000  # 90 days
            })

            result = await container.create_item(analytics_data)
            logger.debug(f"Stored analytics data: {result['id']}")
            return result['id']

        except Exception as e:
            logger.error(f"Error storing analytics data: {str(e)}")
            raise

    async def query_analytics(self, metric_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """Query analytics data"""
        if not self._initialized:
            await self.initialize()

        try:
            container = self.database.get_container_client('analytics')

            query = f"""
                SELECT * FROM c
                WHERE c.metric_type = @metric_type
                AND c.created_at >= DateTimeAdd('day', -{days}, GetCurrentDateTime())
                ORDER BY c.created_at DESC
            """

            items = []
            async for item in container.query_items(
                query,
                parameters=[{"name": "@metric_type", "value": metric_type}],
                max_item_count=1000
            ):
                items.append(item)

            logger.info(f"Retrieved {len(items)} analytics records")
            return items

        except Exception as e:
            logger.error(f"Error querying analytics: {str(e)}")
            raise

    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Cleanup expired data (if TTL is not working)"""
        if not self._initialized:
            await self.initialize()

        cleanup_stats = {}

        try:
            for container_name in self.containers.keys():
                container = self.database.get_container_client(container_name)

                # Query for expired items
                query = """
                    SELECT c.id, c._ts FROM c
                    WHERE c._ts < DateTimeToTimestamp(DateTimeAdd('day', -30, GetCurrentDateTime()))
                """

                expired_items = []
                async for item in container.query_items(query, max_item_count=1000):
                    expired_items.append(item['id'])

                # Delete expired items
                deleted_count = 0
                for item_id in expired_items:
                    try:
                        await container.delete_item(item_id, partition_key=item_id)
                        deleted_count += 1
                    except:
                        pass

                cleanup_stats[container_name] = deleted_count

            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise