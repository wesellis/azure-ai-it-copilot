"""
Database Query Optimization and Performance Monitoring
Advanced database performance optimization with query analysis and monitoring
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from contextlib import asynccontextmanager
import json
import hashlib
from collections import defaultdict, deque
from enum import Enum
import weakref
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from sqlalchemy.engine.events import event
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type enumeration"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRANSACTION = "transaction"
    DDL = "ddl"


@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    query_type: QueryType
    sql_hash: str
    execution_time: float
    rows_examined: int = 0
    rows_returned: int = 0
    memory_usage: int = 0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    connection_id: Optional[str] = None
    table_names: List[str] = field(default_factory=list)
    index_usage: List[str] = field(default_factory=list)


@dataclass
class QueryStats:
    """Aggregated query statistics"""
    total_queries: int = 0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    slow_queries: int = 0
    cache_hit_rate: float = 0.0
    errors: int = 0
    by_type: Dict[QueryType, int] = field(default_factory=lambda: defaultdict(int))
    by_table: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    size: int = 0
    checked_in: int = 0
    checked_out: int = 0
    overflow: int = 0
    invalid: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    avg_checkout_time: float = 0.0
    max_checkout_time: float = 0.0


class QueryOptimizer:
    """Database query optimization and monitoring system"""

    def __init__(self, slow_query_threshold: float = 100.0,
                 enable_explain: bool = True,
                 max_history_size: int = 10000):
        self.slow_query_threshold = slow_query_threshold  # milliseconds
        self.enable_explain = enable_explain
        self.max_history_size = max_history_size

        # Query tracking
        self._query_history: deque = deque(maxlen=max_history_size)
        self._query_cache: Dict[str, Any] = {}
        self._execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._slow_queries: List[QueryMetrics] = []

        # Connection pool monitoring
        self._pool_stats = ConnectionPoolStats()
        self._connection_timings: Dict[str, float] = {}

        # Query pattern analysis
        self._query_patterns: Dict[str, int] = defaultdict(int)
        self._table_access_patterns: Dict[str, List[datetime]] = defaultdict(list)

        # Performance alerts
        self._alert_callbacks: List[Callable] = []

        # Statistics
        self._stats = QueryStats()
        self._start_time = time.time()

    def register_alert_callback(self, callback: Callable):
        """Register callback for performance alerts"""
        self._alert_callbacks.append(callback)

    async def monitor_query(self, query: str, params: Dict[str, Any] = None) -> Callable:
        """Monitor query execution with performance tracking"""
        query_hash = self._hash_query(query)
        query_type = self._detect_query_type(query)
        table_names = self._extract_table_names(query)

        @asynccontextmanager
        async def query_monitor():
            start_time = time.time()
            query_id = f"{query_hash}_{int(start_time * 1000)}"

            try:
                # Pre-execution setup
                await self._pre_execution_hook(query_id, query, params)

                yield query_id

                # Post-execution analysis
                execution_time = (time.time() - start_time) * 1000  # ms
                await self._post_execution_hook(
                    query_id, query_hash, query_type,
                    execution_time, table_names
                )

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                await self._handle_query_error(
                    query_id, query_hash, query_type,
                    execution_time, str(e)
                )
                raise

        return query_monitor()

    async def _pre_execution_hook(self, query_id: str, query: str, params: Dict[str, Any]):
        """Pre-execution analysis and preparation"""
        # Check query cache
        query_hash = self._hash_query(query)
        if query_hash in self._query_cache:
            logger.debug(f"Query cache hit for {query_id}")

        # Pattern analysis
        pattern = self._extract_query_pattern(query)
        self._query_patterns[pattern] += 1

        # Explain plan analysis (if enabled)
        if self.enable_explain and query.strip().lower().startswith('select'):
            await self._analyze_explain_plan(query, params)

    async def _post_execution_hook(self, query_id: str, query_hash: str,
                                 query_type: QueryType, execution_time: float,
                                 table_names: List[str]):
        """Post-execution analysis and metrics collection"""

        # Create metrics record
        metrics = QueryMetrics(
            query_id=query_id,
            query_type=query_type,
            sql_hash=query_hash,
            execution_time=execution_time,
            table_names=table_names,
            cache_hit=query_hash in self._query_cache
        )

        # Store metrics
        self._query_history.append(metrics)
        self._execution_times[query_hash].append(execution_time)

        # Update statistics
        await self._update_stats(metrics)

        # Check for slow queries
        if execution_time > self.slow_query_threshold:
            await self._handle_slow_query(metrics)

        # Update table access patterns
        for table_name in table_names:
            self._table_access_patterns[table_name].append(datetime.utcnow())
            # Keep only last 1000 accesses per table
            if len(self._table_access_patterns[table_name]) > 1000:
                self._table_access_patterns[table_name] = \
                    self._table_access_patterns[table_name][-1000:]

    async def _handle_query_error(self, query_id: str, query_hash: str,
                                query_type: QueryType, execution_time: float,
                                error: str):
        """Handle query execution errors"""
        logger.error(f"Query error for {query_id}: {error}")
        self._stats.errors += 1

        # Alert on errors
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback({
                        'type': 'query_error',
                        'query_id': query_id,
                        'error': error,
                        'execution_time': execution_time,
                        'timestamp': datetime.utcnow()
                    })
                else:
                    callback({
                        'type': 'query_error',
                        'query_id': query_id,
                        'error': error,
                        'execution_time': execution_time,
                        'timestamp': datetime.utcnow()
                    })
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _handle_slow_query(self, metrics: QueryMetrics):
        """Handle slow query detection and analysis"""
        self._slow_queries.append(metrics)
        self._stats.slow_queries += 1

        # Keep only last 100 slow queries
        if len(self._slow_queries) > 100:
            self._slow_queries = self._slow_queries[-100:]

        logger.warning(
            f"Slow query detected: {metrics.query_id} "
            f"({metrics.execution_time:.2f}ms)"
        )

        # Alert on slow queries
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback({
                        'type': 'slow_query',
                        'query_id': metrics.query_id,
                        'execution_time': metrics.execution_time,
                        'threshold': self.slow_query_threshold,
                        'table_names': metrics.table_names,
                        'timestamp': metrics.timestamp
                    })
                else:
                    callback({
                        'type': 'slow_query',
                        'query_id': metrics.query_id,
                        'execution_time': metrics.execution_time,
                        'threshold': self.slow_query_threshold,
                        'table_names': metrics.table_names,
                        'timestamp': metrics.timestamp
                    })
            except Exception as e:
                logger.error(f"Error in slow query alert callback: {e}")

    async def _analyze_explain_plan(self, query: str, params: Dict[str, Any]):
        """Analyze query execution plan for optimization opportunities"""
        try:
            # This would typically connect to the database and run EXPLAIN
            # For now, we'll do pattern-based analysis

            # Check for common anti-patterns
            query_lower = query.lower()

            # Missing WHERE clause on large tables
            if 'select' in query_lower and 'where' not in query_lower:
                logger.warning(f"Query without WHERE clause detected: {query[:100]}...")

            # SELECT * usage
            if 'select *' in query_lower:
                logger.warning(f"SELECT * usage detected: {query[:100]}...")

            # Subqueries that could be JOINs
            if query_lower.count('select') > 1:
                logger.info(f"Subquery detected, consider JOIN optimization: {query[:100]}...")

        except Exception as e:
            logger.error(f"Error analyzing explain plan: {e}")

    async def _update_stats(self, metrics: QueryMetrics):
        """Update aggregated statistics"""
        self._stats.total_queries += 1

        # Update execution time statistics
        times = [m.execution_time for m in self._query_history if m.query_type == metrics.query_type]
        if times:
            self._stats.avg_execution_time = statistics.mean(times)
            self._stats.min_execution_time = min(times)
            self._stats.max_execution_time = max(times)

            if len(times) >= 20:  # Need sufficient data for percentiles
                self._stats.p95_execution_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
                self._stats.p99_execution_time = statistics.quantiles(times, n=100)[98]  # 99th percentile

        # Update by type statistics
        self._stats.by_type[metrics.query_type] += 1

        # Update by table statistics
        for table_name in metrics.table_names:
            self._stats.by_table[table_name] += 1

        # Update cache hit rate
        cache_hits = sum(1 for m in self._query_history if m.cache_hit)
        self._stats.cache_hit_rate = (cache_hits / len(self._query_history) * 100) if self._query_history else 0

    def _hash_query(self, query: str) -> str:
        """Generate hash for query normalization"""
        # Normalize query by removing parameters and whitespace
        normalized = ' '.join(query.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from SQL statement"""
        query_lower = query.strip().lower()

        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith(('begin', 'commit', 'rollback')):
            return QueryType.TRANSACTION
        else:
            return QueryType.DDL

    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        # Simplified table name extraction
        tables = []
        query_lower = query.lower()

        # Look for FROM and JOIN clauses
        import re

        # FROM clause
        from_match = re.search(r'\bfrom\s+(\w+)', query_lower)
        if from_match:
            tables.append(from_match.group(1))

        # JOIN clauses
        join_matches = re.findall(r'\bjoin\s+(\w+)', query_lower)
        tables.extend(join_matches)

        # UPDATE and INSERT statements
        update_match = re.search(r'\bupdate\s+(\w+)', query_lower)
        if update_match:
            tables.append(update_match.group(1))

        insert_match = re.search(r'\binto\s+(\w+)', query_lower)
        if insert_match:
            tables.append(insert_match.group(1))

        return list(set(tables))  # Remove duplicates

    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for analysis"""
        # Normalize query to pattern
        import re

        # Replace values with placeholders
        pattern = re.sub(r"'[^']*'", "'?'", query)
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        pattern = re.sub(r'\s+', ' ', pattern)

        return pattern.strip()

    def monitor_connection_pool(self, pool):
        """Monitor database connection pool performance"""
        if hasattr(pool, 'size'):
            self._pool_stats.size = pool.size()
        if hasattr(pool, 'checked_in'):
            self._pool_stats.checked_in = pool.checkedin()
        if hasattr(pool, 'checked_out'):
            self._pool_stats.checked_out = pool.checkedout()
        if hasattr(pool, 'overflow'):
            self._pool_stats.overflow = pool.overflow()
        if hasattr(pool, 'invalid'):
            self._pool_stats.invalid = pool.invalid()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        uptime = time.time() - self._start_time

        return {
            'uptime_seconds': uptime,
            'query_stats': {
                'total_queries': self._stats.total_queries,
                'queries_per_second': self._stats.total_queries / uptime if uptime > 0 else 0,
                'avg_execution_time_ms': self._stats.avg_execution_time,
                'p95_execution_time_ms': self._stats.p95_execution_time,
                'p99_execution_time_ms': self._stats.p99_execution_time,
                'slow_queries': self._stats.slow_queries,
                'cache_hit_rate': self._stats.cache_hit_rate,
                'errors': self._stats.errors,
                'by_type': dict(self._stats.by_type),
                'by_table': dict(self._stats.by_table)
            },
            'connection_pool': {
                'size': self._pool_stats.size,
                'checked_in': self._pool_stats.checked_in,
                'checked_out': self._pool_stats.checked_out,
                'utilization': (self._pool_stats.checked_out / self._pool_stats.size * 100)
                             if self._pool_stats.size > 0 else 0
            },
            'top_patterns': dict(sorted(
                self._query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'recent_slow_queries': [
                {
                    'query_id': q.query_id,
                    'execution_time_ms': q.execution_time,
                    'table_names': q.table_names,
                    'timestamp': q.timestamp.isoformat()
                }
                for q in self._slow_queries[-10:]
            ]
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on collected metrics"""
        recommendations = []

        # High cache miss rate
        if self._stats.cache_hit_rate < 70:
            recommendations.append({
                'type': 'caching',
                'priority': 'high',
                'title': 'Low cache hit rate',
                'description': f'Cache hit rate is {self._stats.cache_hit_rate:.1f}%. Consider implementing query result caching.',
                'impact': 'Medium performance improvement'
            })

        # Slow average query time
        if self._stats.avg_execution_time > 50:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'High average query time',
                'description': f'Average query time is {self._stats.avg_execution_time:.1f}ms. Review query optimization and indexing.',
                'impact': 'High performance improvement'
            })

        # High slow query count
        slow_query_rate = (self._stats.slow_queries / self._stats.total_queries * 100) if self._stats.total_queries > 0 else 0
        if slow_query_rate > 5:
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'title': 'High slow query rate',
                'description': f'{slow_query_rate:.1f}% of queries are slow. Review and optimize frequent slow queries.',
                'impact': 'High performance improvement'
            })

        # High connection pool utilization
        pool_utilization = (self._pool_stats.checked_out / self._pool_stats.size * 100) if self._pool_stats.size > 0 else 0
        if pool_utilization > 80:
            recommendations.append({
                'type': 'infrastructure',
                'priority': 'medium',
                'title': 'High connection pool utilization',
                'description': f'Connection pool utilization is {pool_utilization:.1f}%. Consider increasing pool size.',
                'impact': 'Medium performance improvement'
            })

        # Frequent table access patterns
        for table, accesses in self._table_access_patterns.items():
            recent_accesses = [a for a in accesses if a > datetime.utcnow() - timedelta(hours=1)]
            if len(recent_accesses) > 100:  # More than 100 accesses per hour
                recommendations.append({
                    'type': 'caching',
                    'priority': 'medium',
                    'title': f'Frequent access to table {table}',
                    'description': f'Table {table} accessed {len(recent_accesses)} times in the last hour. Consider caching.',
                    'impact': 'Medium performance improvement'
                })

        return recommendations

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary()
        recommendations = self.get_optimization_recommendations()

        return {
            'report_timestamp': datetime.utcnow().isoformat(),
            'summary': summary,
            'recommendations': recommendations,
            'health_score': self._calculate_health_score(),
            'trend_analysis': self._analyze_performance_trends()
        }

    def _calculate_health_score(self) -> int:
        """Calculate database performance health score (0-100)"""
        score = 100

        # Penalize for slow queries
        slow_query_rate = (self._stats.slow_queries / self._stats.total_queries * 100) if self._stats.total_queries > 0 else 0
        score -= min(slow_query_rate * 2, 30)

        # Penalize for low cache hit rate
        if self._stats.cache_hit_rate < 70:
            score -= (70 - self._stats.cache_hit_rate) / 2

        # Penalize for high average execution time
        if self._stats.avg_execution_time > 50:
            score -= min((self._stats.avg_execution_time - 50) / 10, 20)

        # Penalize for errors
        error_rate = (self._stats.errors / self._stats.total_queries * 100) if self._stats.total_queries > 0 else 0
        score -= min(error_rate * 5, 20)

        return max(int(score), 0)

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self._query_history) < 2:
            return {'status': 'insufficient_data'}

        # Analyze recent vs historical performance
        recent_queries = [q for q in self._query_history if q.timestamp > datetime.utcnow() - timedelta(hours=1)]
        historical_queries = [q for q in self._query_history if q.timestamp <= datetime.utcnow() - timedelta(hours=1)]

        if not recent_queries or not historical_queries:
            return {'status': 'insufficient_data'}

        recent_avg = statistics.mean([q.execution_time for q in recent_queries])
        historical_avg = statistics.mean([q.execution_time for q in historical_queries])

        trend = 'improving' if recent_avg < historical_avg else 'degrading' if recent_avg > historical_avg else 'stable'
        change_percent = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0

        return {
            'status': 'analyzed',
            'trend': trend,
            'change_percent': change_percent,
            'recent_avg_ms': recent_avg,
            'historical_avg_ms': historical_avg,
            'recent_queries_count': len(recent_queries),
            'historical_queries_count': len(historical_queries)
        }


# Global query optimizer instance
query_optimizer = QueryOptimizer()


# Decorator for automatic query monitoring
def monitor_db_query(slow_threshold: float = 100.0):
    """Decorator to automatically monitor database queries"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract query information if available
            query_info = getattr(func, '__query_info__', {})

            async with query_optimizer.monitor_query(
                query_info.get('sql', func.__name__),
                query_info.get('params', {})
            ):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            query_info = getattr(func, '__query_info__', {})

            async def monitored_execution():
                async with query_optimizer.monitor_query(
                    query_info.get('sql', func.__name__),
                    query_info.get('params', {})
                ):
                    return func(*args, **kwargs)

            return loop.run_until_complete(monitored_execution())

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Connection pool monitoring
class MonitoredQueuePool(QueuePool):
    """Enhanced QueuePool with monitoring capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkout_times = {}

    def _do_get(self):
        start_time = time.time()
        conn = super()._do_get()
        checkout_time = time.time() - start_time

        # Update connection pool stats
        query_optimizer._pool_stats.pool_hits += 1
        query_optimizer._pool_stats.avg_checkout_time = (
            query_optimizer._pool_stats.avg_checkout_time + checkout_time
        ) / 2
        query_optimizer._pool_stats.max_checkout_time = max(
            query_optimizer._pool_stats.max_checkout_time,
            checkout_time
        )

        return conn

    def _do_return_conn(self, conn):
        result = super()._do_return_conn(conn)
        query_optimizer.monitor_connection_pool(self)
        return result


# Utility functions
async def get_query_performance_report() -> Dict[str, Any]:
    """Get comprehensive query performance report"""
    return await query_optimizer.generate_performance_report()


async def get_optimization_recommendations() -> List[Dict[str, Any]]:
    """Get database optimization recommendations"""
    return query_optimizer.get_optimization_recommendations()


def register_performance_alert(callback: Callable):
    """Register callback for database performance alerts"""
    query_optimizer.register_alert_callback(callback)