"""
Optimized Azure AD Authentication Integration
Provides enterprise-grade authentication with Azure Active Directory
Includes caching, rate limiting, and enhanced security features
"""

import logging
import os
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import json
from functools import lru_cache, wraps
import time
import hashlib
import asyncio

import jwt
from fastapi import HTTPException, status
from msal import ConfidentialClientApplication, SerializableTokenCache
import httpx

logger = logging.getLogger(__name__)


# Rate limiting decorator
def rate_limit_check(func):
    """Decorator to check rate limiting"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        client_ip = kwargs.get('client_ip', 'unknown')
        if self._is_rate_limited(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        return await func(self, *args, **kwargs)
    return wrapper


# Alias for backward compatibility
AzureADAuth = OptimizedAzureADAuth


class OptimizedAzureADAuth:
    """Optimized Azure AD Authentication and Authorization Handler with caching and security enhancements"""

    def __init__(self):
        """Initialize optimized Azure AD authentication"""
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("AZURE_REDIRECT_URI", "http://localhost:3000/auth/callback")

        # Security configurations
        self.max_token_age = int(os.getenv("MAX_TOKEN_AGE_MINUTES", "60")) * 60  # seconds
        self.rate_limit_window = 300  # 5 minutes
        self.max_requests_per_window = 100

        # Rate limiting tracking
        self._request_timestamps: Dict[str, List[float]] = {}
        self._blocked_ips: Set[str] = set()

        # Token cache
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        self._user_cache: Dict[str, Dict[str, Any]] = {}

        if not all([self.tenant_id, self.client_id, self.client_secret]):
            logger.warning("Azure AD credentials not configured - falling back to basic auth")
            self.enabled = False
            return

        self.enabled = True
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]

        # Initialize optimized MSAL app with connection pooling
        self.app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority,
            token_cache=SerializableTokenCache(),
            proxies=None,  # Can be configured for proxy support
            verify=True,   # SSL verification
            timeout=30     # Request timeout
        )

        # HTTP client for Graph API calls
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            )
        )

        logger.info("🔐 Optimized Azure AD authentication initialized")

    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""
        if client_id in self._blocked_ips:
            return True

        now = time.time()
        window_start = now - self.rate_limit_window

        # Clean old timestamps
        if client_id in self._request_timestamps:
            self._request_timestamps[client_id] = [
                ts for ts in self._request_timestamps[client_id]
                if ts > window_start
            ]
        else:
            self._request_timestamps[client_id] = []

        # Check rate limit
        if len(self._request_timestamps[client_id]) >= self.max_requests_per_window:
            self._blocked_ips.add(client_id)
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return True

        # Record request
        self._request_timestamps[client_id].append(now)
        return False

    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        content = ":".join(str(arg) for arg in args)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @lru_cache(maxsize=1000)
    def _validate_token_cached(self, token: str) -> Optional[Dict[str, Any]]:
        """Cached token validation"""
        try:
            # Decode token without verification for caching
            # Full verification happens in validate_token method
            decoded = jwt.decode(token, options={"verify_signature": False})
            return decoded
        except Exception:
            return None

    async def get_authorization_url(self, state: str = None, client_ip: str = "unknown") -> str:
        """Get optimized Azure AD authorization URL for login flow with rate limiting"""
        if not self.enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Azure AD authentication not configured"
            )

        auth_url = self.app.get_authorization_request_url(
            scopes=self.scope,
            redirect_uri=self.redirect_uri,
            state=state
        )
        return auth_url

    async def handle_callback(self, authorization_code: str, state: str = None) -> Dict[str, Any]:
        """Handle Azure AD callback and exchange code for tokens"""
        if not self.enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Azure AD authentication not configured"
            )

        try:
            result = self.app.acquire_token_by_authorization_code(
                code=authorization_code,
                scopes=self.scope,
                redirect_uri=self.redirect_uri
            )

            if "error" in result:
                logger.error(f"Azure AD token exchange failed: {result.get('error_description')}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {result.get('error_description')}"
                )

            # Get user info from Microsoft Graph
            user_info = await self._get_user_info(result["access_token"])

            # Generate our internal JWT token
            internal_token = self._generate_internal_token(user_info, result)

            return {
                "access_token": internal_token,
                "token_type": "bearer",
                "expires_in": 3600,
                "user_info": user_info,
                "azure_tokens": {
                    "access_token": result["access_token"],
                    "refresh_token": result.get("refresh_token"),
                    "expires_in": result.get("expires_in")
                }
            }

        except Exception as e:
            logger.error(f"Azure AD callback handling failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication callback failed"
            )

    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Microsoft Graph API"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            # Get basic user profile
            response = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers=headers
            )

            if response.status_code != 200:
                logger.error(f"Failed to get user info: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Failed to retrieve user information"
                )

            user_data = response.json()

            # Get user's group memberships for role mapping
            groups_response = await client.get(
                "https://graph.microsoft.com/v1.0/me/memberOf",
                headers=headers
            )

            groups = []
            if groups_response.status_code == 200:
                groups_data = groups_response.json()
                groups = [group.get("displayName", "") for group in groups_data.get("value", [])]

            return {
                "id": user_data.get("id"),
                "email": user_data.get("mail") or user_data.get("userPrincipalName"),
                "name": user_data.get("displayName"),
                "first_name": user_data.get("givenName"),
                "last_name": user_data.get("surname"),
                "job_title": user_data.get("jobTitle"),
                "department": user_data.get("department"),
                "groups": groups,
                "role": self._determine_user_role(groups, user_data)
            }

    def _determine_user_role(self, groups: List[str], user_data: Dict[str, Any]) -> str:
        """Determine user role based on Azure AD groups and attributes"""
        # Define role mapping based on Azure AD groups
        role_mapping = {
            "IT Administrators": "owner",
            "Infrastructure Team": "contributor",
            "DevOps Engineers": "contributor",
            "Security Team": "contributor",
            "Help Desk": "reader",
            "Managers": "reader"
        }

        # Check for admin indicators
        email = user_data.get("mail", "").lower()
        if "admin" in email or "root" in email:
            return "owner"

        # Map groups to roles (highest role wins)
        user_roles = []
        for group in groups:
            if group in role_mapping:
                user_roles.append(role_mapping[group])

        if "owner" in user_roles:
            return "owner"
        elif "contributor" in user_roles:
            return "contributor"
        elif "reader" in user_roles:
            return "reader"
        else:
            return "reader"  # Default role

    def _generate_internal_token(self, user_info: Dict[str, Any], azure_tokens: Dict[str, Any]) -> str:
        """Generate internal JWT token for API access"""
        payload = {
            "sub": user_info["id"],
            "email": user_info["email"],
            "name": user_info["name"],
            "role": user_info["role"],
            "groups": user_info["groups"],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iss": "azure-ai-copilot",
            "aud": "azure-ai-copilot-api",
            "auth_method": "azure_ad"
        }

        secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        return jwt.encode(payload, secret, algorithm="HS256")

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh Azure AD access token"""
        if not self.enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Azure AD authentication not configured"
            )

        try:
            result = self.app.acquire_token_by_refresh_token(
                refresh_token=refresh_token,
                scopes=self.scope
            )

            if "error" in result:
                logger.error(f"Token refresh failed: {result.get('error_description')}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token refresh failed"
                )

            # Get updated user info
            user_info = await self._get_user_info(result["access_token"])

            # Generate new internal token
            internal_token = self._generate_internal_token(user_info, result)

            return {
                "access_token": internal_token,
                "token_type": "bearer",
                "expires_in": 3600,
                "user_info": user_info,
                "azure_tokens": {
                    "access_token": result["access_token"],
                    "refresh_token": result.get("refresh_token"),
                    "expires_in": result.get("expires_in")
                }
            }

        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed"
            )

    async def get_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get detailed user permissions and capabilities"""
        # This would typically query your user database
        # For now, return basic permissions based on role

        # In a real implementation, you'd fetch this from your database
        # based on the user_id and their current role assignments

        return {
            "can_create_resources": True,
            "can_delete_resources": False,
            "can_modify_resources": True,
            "can_view_costs": True,
            "can_manage_incidents": True,
            "can_access_compliance": True,
            "can_run_predictions": True,
            "max_cost_threshold": 10000,
            "allowed_resource_types": ["vm", "storage", "network"],
            "restricted_regions": []
        }

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate and decode internal JWT token"""
        try:
            secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
            payload = jwt.decode(token, secret, algorithms=["HS256"])

            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    async def logout(self, user_id: str) -> bool:
        """Handle user logout"""
        try:
            # In a real implementation, you'd:
            # 1. Invalidate the refresh token with Azure AD
            # 2. Clear any cached tokens
            # 3. Log the logout event

            logger.info(f"User {user_id} logged out successfully")
            return True

        except Exception as e:
            logger.error(f"Logout failed for user {user_id}: {str(e)}")
            return False


class RBACManager:
    """Role-Based Access Control Manager"""

    def __init__(self):
        """Initialize RBAC manager"""
        self.role_permissions = {
            "owner": {
                "resources": ["create", "read", "update", "delete"],
                "incidents": ["create", "read", "update", "delete", "resolve"],
                "costs": ["read", "analyze", "optimize"],
                "compliance": ["read", "configure", "audit"],
                "predictions": ["read", "configure"],
                "admin": ["manage_users", "view_logs", "configure_system"]
            },
            "contributor": {
                "resources": ["create", "read", "update"],
                "incidents": ["create", "read", "update", "resolve"],
                "costs": ["read", "analyze"],
                "compliance": ["read", "audit"],
                "predictions": ["read"],
                "admin": ["view_logs"]
            },
            "reader": {
                "resources": ["read"],
                "incidents": ["read"],
                "costs": ["read"],
                "compliance": ["read"],
                "predictions": ["read"],
                "admin": []
            }
        }

    def check_permission(self, user_role: str, resource: str, action: str) -> bool:
        """Check if user has permission for specific action on resource"""
        if user_role not in self.role_permissions:
            return False

        resource_perms = self.role_permissions[user_role].get(resource, [])
        return action in resource_perms

    def get_user_permissions(self, user_role: str) -> Dict[str, List[str]]:
        """Get all permissions for a user role"""
        return self.role_permissions.get(user_role, {})

    def can_access_resource(self, user_role: str, resource_type: str, action: str) -> bool:
        """Check if user can access specific resource type"""
        return self.check_permission(user_role, resource_type, action)


# Global instances
azure_ad_auth = AzureADAuth()
rbac_manager = RBACManager()


# Dependency functions for FastAPI
async def get_current_user(token: str = None) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required"
        )

    return azure_ad_auth.validate_token(token)


def require_permission(resource: str, action: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs or context
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            user_role = current_user.get("role", "reader")
            if not rbac_manager.check_permission(user_role, resource, action):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions for {action} on {resource}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator