"""
Enterprise-grade rate limiting for AI API calls with advanced features
Supports multiple strategies, user tiers, and compliance logging
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of rate limits"""
    REQUEST_COUNT = "request_count"
    TOKEN_USAGE = "token_usage"
    COST_LIMIT = "cost_limit"
    CONCURRENT = "concurrent"


class UserTier(Enum):
    """User subscription tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_type: LimitType
    limit: int
    window_seconds: int
    burst_allowance: int = 0
    description: str = ""


@dataclass
class RateLimitEvent:
    """Rate limit event for audit logging"""
    user_id: str
    endpoint: str
    limit_type: LimitType
    current_usage: int
    limit: int
    action: str  # "allowed", "throttled", "blocked"
    timestamp: float
    metadata: Dict[str, Any]


class AdvancedRateLimiter:
    """Enterprise-grade rate limiter with multiple strategies"""

    def __init__(self):
        self._limits: Dict[UserTier, List[RateLimit]] = {}
        self._usage_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self._concurrent_requests: Dict[str, int] = defaultdict(int)
        self._events: List[RateLimitEvent] = []
        self._lock = asyncio.Lock()

        # Configure default limits
        self._setup_default_limits()

    def _setup_default_limits(self):
        """Setup default rate limits for each tier"""

        # Free tier - very restrictive
        self._limits[UserTier.FREE] = [
            RateLimit(LimitType.REQUEST_COUNT, 10, 3600, 2, "10 requests per hour"),
            RateLimit(LimitType.TOKEN_USAGE, 5000, 3600, 500, "5K tokens per hour"),
            RateLimit(LimitType.COST_LIMIT, 100, 86400, 10, "$1.00 per day"),
            RateLimit(LimitType.CONCURRENT, 1, 0, 0, "1 concurrent request"),
        ]

        # Starter tier - basic usage
        self._limits[UserTier.STARTER] = [
            RateLimit(LimitType.REQUEST_COUNT, 100, 3600, 20, "100 requests per hour"),
            RateLimit(LimitType.TOKEN_USAGE, 50000, 3600, 5000, "50K tokens per hour"),
            RateLimit(LimitType.COST_LIMIT, 1000, 86400, 100, "$10.00 per day"),
            RateLimit(LimitType.CONCURRENT, 3, 0, 1, "3 concurrent requests"),
        ]

        # Professional tier - regular business use
        self._limits[UserTier.PROFESSIONAL] = [
            RateLimit(LimitType.REQUEST_COUNT, 1000, 3600, 100, "1K requests per hour"),
            RateLimit(LimitType.TOKEN_USAGE, 500000, 3600, 50000, "500K tokens per hour"),
            RateLimit(LimitType.COST_LIMIT, 10000, 86400, 1000, "$100.00 per day"),
            RateLimit(LimitType.CONCURRENT, 10, 0, 2, "10 concurrent requests"),
        ]

        # Enterprise tier - high volume
        self._limits[UserTier.ENTERPRISE] = [
            RateLimit(LimitType.REQUEST_COUNT, 10000, 3600, 1000, "10K requests per hour"),
            RateLimit(LimitType.TOKEN_USAGE, 5000000, 3600, 500000, "5M tokens per hour"),
            RateLimit(LimitType.COST_LIMIT, 100000, 86400, 10000, "$1000.00 per day"),
            RateLimit(LimitType.CONCURRENT, 50, 0, 10, "50 concurrent requests"),
        ]

    async def check_limits(self,
                          user_id: str,
                          user_tier: UserTier,
                          endpoint: str,
                          request_cost: float = 0.01,
                          token_count: int = 100) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limits

        Returns:
            (allowed: bool, info: dict)
        """
        async with self._lock:
            current_time = time.time()

            # Get limits for user tier
            limits = self._limits.get(user_tier, self._limits[UserTier.FREE])

            # Check each limit type
            limit_results = {}
            overall_allowed = True

            for limit in limits:
                allowed, usage, remaining = await self._check_single_limit(
                    user_id, endpoint, limit, current_time, request_cost, token_count
                )

                limit_results[limit.limit_type.value] = {
                    "allowed": allowed,
                    "usage": usage,
                    "limit": limit.limit,
                    "remaining": remaining,
                    "window_seconds": limit.window_seconds,
                    "description": limit.description
                }

                if not allowed:
                    overall_allowed = False

                    # Log rate limit event
                    event = RateLimitEvent(
                        user_id=user_id,
                        endpoint=endpoint,
                        limit_type=limit.limit_type,
                        current_usage=usage,
                        limit=limit.limit,
                        action="blocked",
                        timestamp=current_time,
                        metadata={
                            "user_tier": user_tier.value,
                            "request_cost": request_cost,
                            "token_count": token_count
                        }
                    )
                    self._events.append(event)

                    logger.warning(f"Rate limit exceeded for user {user_id}: {limit.limit_type.value}")

            # If allowed, increment usage
            if overall_allowed:
                await self._increment_usage(user_id, endpoint, limits, current_time, request_cost, token_count)

                # Log allowed event
                event = RateLimitEvent(
                    user_id=user_id,
                    endpoint=endpoint,
                    limit_type=LimitType.REQUEST_COUNT,
                    current_usage=limit_results[LimitType.REQUEST_COUNT.value]["usage"] + 1,
                    limit=limit_results[LimitType.REQUEST_COUNT.value]["limit"],
                    action="allowed",
                    timestamp=current_time,
                    metadata={
                        "user_tier": user_tier.value,
                        "request_cost": request_cost,
                        "token_count": token_count
                    }
                )
                self._events.append(event)

            return overall_allowed, {
                "allowed": overall_allowed,
                "limits": limit_results,
                "user_tier": user_tier.value,
                "timestamp": current_time
            }

    async def _check_single_limit(self,
                                 user_id: str,
                                 endpoint: str,
                                 limit: RateLimit,
                                 current_time: float,
                                 request_cost: float,
                                 token_count: int) -> Tuple[bool, int, int]:
        """Check a single rate limit"""

        if limit.limit_type == LimitType.CONCURRENT:
            return self._check_concurrent_limit(user_id, limit)

        # For time-window based limits
        window_key = f"{user_id}:{endpoint}:{limit.limit_type.value}"
        window = self._usage_windows[window_key]

        # Clean old entries outside the window
        cutoff_time = current_time - limit.window_seconds
        while window and window[0][0] < cutoff_time:
            window.popleft()

        # Calculate current usage
        if limit.limit_type == LimitType.REQUEST_COUNT:
            current_usage = len(window)
            increment_amount = 1
        elif limit.limit_type == LimitType.TOKEN_USAGE:
            current_usage = sum(entry[1] for entry in window)
            increment_amount = token_count
        elif limit.limit_type == LimitType.COST_LIMIT:
            current_usage = sum(entry[1] for entry in window)
            increment_amount = int(request_cost * 100)  # Convert to cents
        else:
            current_usage = len(window)
            increment_amount = 1

        # Check if adding this request would exceed limit
        effective_limit = limit.limit + limit.burst_allowance
        remaining = effective_limit - current_usage
        allowed = (current_usage + increment_amount) <= effective_limit

        return allowed, current_usage, max(0, remaining)

    def _check_concurrent_limit(self, user_id: str, limit: RateLimit) -> Tuple[bool, int, int]:
        """Check concurrent request limit"""
        current_concurrent = self._concurrent_requests.get(user_id, 0)
        effective_limit = limit.limit + limit.burst_allowance
        remaining = effective_limit - current_concurrent
        allowed = current_concurrent < effective_limit

        return allowed, current_concurrent, max(0, remaining)

    async def _increment_usage(self,
                              user_id: str,
                              endpoint: str,
                              limits: List[RateLimit],
                              current_time: float,
                              request_cost: float,
                              token_count: int):
        """Increment usage counters for all limit types"""

        for limit in limits:
            if limit.limit_type == LimitType.CONCURRENT:
                self._concurrent_requests[user_id] += 1
                continue

            window_key = f"{user_id}:{endpoint}:{limit.limit_type.value}"
            window = self._usage_windows[window_key]

            if limit.limit_type == LimitType.REQUEST_COUNT:
                window.append((current_time, 1))
            elif limit.limit_type == LimitType.TOKEN_USAGE:
                window.append((current_time, token_count))
            elif limit.limit_type == LimitType.COST_LIMIT:
                window.append((current_time, int(request_cost * 100)))

    async def release_concurrent(self, user_id: str):
        """Release a concurrent request slot"""
        async with self._lock:
            if self._concurrent_requests[user_id] > 0:
                self._concurrent_requests[user_id] -= 1

    def get_usage_stats(self, user_id: str, user_tier: UserTier) -> Dict[str, Any]:
        """Get current usage statistics for a user"""
        current_time = time.time()
        limits = self._limits.get(user_tier, self._limits[UserTier.FREE])

        stats = {}
        for limit in limits:
            window_key = f"{user_id}:*:{limit.limit_type.value}"

            if limit.limit_type == LimitType.CONCURRENT:
                current_usage = self._concurrent_requests.get(user_id, 0)
            else:
                # Aggregate across all endpoints for this user and limit type
                current_usage = 0
                for key, window in self._usage_windows.items():
                    if key.startswith(f"{user_id}:") and key.endswith(f":{limit.limit_type.value}"):
                        cutoff_time = current_time - limit.window_seconds
                        valid_entries = [entry for entry in window if entry[0] >= cutoff_time]

                        if limit.limit_type == LimitType.REQUEST_COUNT:
                            current_usage += len(valid_entries)
                        else:
                            current_usage += sum(entry[1] for entry in valid_entries)

            stats[limit.limit_type.value] = {
                "current": current_usage,
                "limit": limit.limit,
                "remaining": max(0, limit.limit - current_usage),
                "window_seconds": limit.window_seconds,
                "description": limit.description
            }

        return stats

    def get_rate_limit_events(self,
                             user_id: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get rate limit events for audit logging"""
        events = self._events

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        # Sort by timestamp, most recent first
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)

        return [asdict(event) for event in events[:limit]]

    def clear_old_events(self, max_age_hours: int = 24):
        """Clear old rate limit events"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        self._events = [e for e in self._events if e.timestamp >= cutoff_time]

    def set_custom_limits(self, user_tier: UserTier, limits: List[RateLimit]):
        """Set custom rate limits for a user tier"""
        self._limits[user_tier] = limits

    def add_custom_limit(self, user_tier: UserTier, limit: RateLimit):
        """Add a custom rate limit to a user tier"""
        if user_tier not in self._limits:
            self._limits[user_tier] = []
        self._limits[user_tier].append(limit)


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()


# Decorator for FastAPI endpoints
def rate_limit(user_tier: UserTier = UserTier.FREE,
               request_cost: float = 0.01,
               token_count: int = 100):
    """Rate limiting decorator for FastAPI endpoints"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from request context
            # This would need to be adapted based on your auth system
            user_id = kwargs.get('user_id', 'anonymous')
            endpoint = f"{func.__module__}.{func.__name__}"

            allowed, info = await rate_limiter.check_limits(
                user_id=user_id,
                user_tier=user_tier,
                endpoint=endpoint,
                request_cost=request_cost,
                token_count=token_count
            )

            if not allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limits": info["limits"],
                        "retry_after": 3600  # Suggest retry after 1 hour
                    }
                )

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                # Release concurrent slot if this was a concurrent-limited request
                await rate_limiter.release_concurrent(user_id)

        return wrapper
    return decorator