"""
FastAPI Server for Azure AI IT Copilot
Provides REST API endpoints for the AI orchestrator
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache, wraps
import time

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from passlib.context import CryptContext
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import logging

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ai_orchestrator.orchestrator import AzureAIOrchestrator, IntentType
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("AzureAIOrchestrator not available - running in standalone mode")
    AzureAIOrchestrator = None
    IntentType = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Metrics with better labels
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'API request duration', buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
command_processing_time = Histogram('command_processing_seconds', 'Command processing time', buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0])
active_connections = Counter('websocket_connections_total', 'Total WebSocket connections')
error_count = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])


# Pydantic models
class CommandRequest(BaseModel):
    """Optimized request model for processing commands"""
    command: str = Field(..., min_length=1, max_length=1000, description="Natural language command to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    auto_approve: bool = Field(default=False, description="Auto-approve operations")
    dry_run: bool = Field(default=False, description="Simulate without executing")
    timeout: Optional[int] = Field(default=300, ge=10, le=3600, description="Command timeout in seconds")

    @validator('command')
    def validate_command(cls, v):
        if not v.strip():
            raise ValueError('Command cannot be empty')
        return v.strip()


class CommandResponse(BaseModel):
    """Response model for command execution"""
    request_id: str
    status: str
    result: Dict[str, Any]
    execution_time: float
    timestamp: str


class ResourceQuery(BaseModel):
    """Optimized query model for resources"""
    resource_type: Optional[str] = Field(None, max_length=100)
    resource_group: Optional[str] = Field(None, max_length=90)  # Azure limit
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    include_metrics: bool = Field(default=False)
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Results offset for pagination")


class IncidentReport(BaseModel):
    """Optimized model for incident reporting"""
    description: str = Field(..., min_length=10, max_length=2000)
    severity: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    affected_resources: List[str] = Field(..., min_items=1, max_items=50)
    symptoms: List[str] = Field(..., min_items=1, max_items=20)
    auto_remediate: bool = Field(default=True)
    priority: Optional[int] = Field(None, ge=1, le=5, description="Priority level 1-5")


class CostAnalysisRequest(BaseModel):
    """Request model for cost analysis"""
    scope: str = Field(default="subscription", pattern="^(subscription|resource_group|resource)$")
    resource_group: Optional[str] = None
    timeframe: str = Field(default="30d")
    optimization_level: str = Field(default="moderate", pattern="^(conservative|moderate|aggressive)$")


class AuthRequest(BaseModel):
    """Authentication request model"""
    username: str
    password: str


class AuthResponse(BaseModel):
    """Authentication response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]


# Optimized Security with caching
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cache for verified tokens to reduce JWT decode overhead
@lru_cache(maxsize=1000)
def _decode_token_cached(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """Cached token decoding to improve performance"""
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except (ExpiredSignatureError, JWTError):
        return None

# Rate limiting helper
def rate_limit(max_calls: int = 100, window_seconds: int = 60):
    """Simple rate limiting decorator"""
    calls = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            # Clean old entries
            calls.update({k: v for k, v in calls.items() if now - v < window_seconds})

            client_id = kwargs.get('request', {}).client.host if 'request' in kwargs else 'unknown'
            if calls.get(client_id, 0) > max_calls:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            calls[client_id] = calls.get(client_id, 0) + 1
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global application state
orchestrator: Optional[AzureAIOrchestrator] = None
active_websockets: List[WebSocket] = []
app_start_time = datetime.now()


class RedisManager:
    """Optimized Redis operations manager with connection pooling"""
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self, redis_url: str) -> bool:
        """Connect to Redis with proper error handling"""
        try:
            self.client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            await self.client.ping()
            self._connected = True
            logger.info("Redis connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            self._connected = False
            return False

    async def disconnect(self):
        """Safely disconnect from Redis"""
        if self.client:
            await self.client.close()
            self._connected = False

    async def safe_operation(self, operation_name: str, operation_func, default_return=None):
        """Execute Redis operations with automatic retry and error handling"""
        if not self._connected or not self.client:
            logger.warning(f"Redis operation '{operation_name}' skipped: Redis not available")
            return default_return

        try:
            return await operation_func(self.client)
        except redis.ConnectionError:
            logger.warning(f"Redis connection lost during '{operation_name}', attempting reconnect...")
            self._connected = False
            return default_return
        except Exception as e:
            logger.warning(f"Redis operation '{operation_name}' failed: {e}")
            return default_return

# Global Redis manager
redis_manager = RedisManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifecycle management"""
    global orchestrator

    # Startup
    logger.info("🚀 Starting Azure AI IT Copilot API")

    # Initialize orchestrator if available
    if AzureAIOrchestrator:
        try:
            orchestrator = AzureAIOrchestrator()
            logger.info("✅ Azure AI Orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize orchestrator: {e}")
            orchestrator = None
    else:
        orchestrator = None
        logger.warning("⚠️ Running without AI Orchestrator")

    # Initialize Redis with connection pooling
    redis_url = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}"
    await redis_manager.connect(redis_url)

    # Initialize metrics collection
    logger.info("📊 Metrics collection initialized")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Azure AI IT Copilot API")
    await redis_manager.disconnect()
    logger.info("✅ Graceful shutdown completed")


@lru_cache()
def get_cors_origins() -> List[str]:
    """Get CORS origins with caching"""
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    if cors_origins == "*":
        return ["*"]
    return [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

# Create optimized FastAPI app
app = FastAPI(
    title="Azure AI IT Copilot API",
    description="Enterprise-grade natural language interface for Azure infrastructure management",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan,
    # Optimize JSON serialization
    default_response_class=None,
)

# Configure optimized CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

# Add security middleware with proper hosts
allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[host.strip() for host in allowed_hosts]
)

# Optimized security headers and request tracking middleware
@app.middleware("http")
async def add_security_and_tracking_headers(request: Request, call_next):
    # Generate request ID for tracing
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Record request start time
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate response time
    process_time = time.time() - start_time

    # Add tracking headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{process_time:.3f}s"

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    # HSTS only in production with HTTPS
    if os.getenv("ENVIRONMENT") == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

    # Record metrics
    request_duration.observe(process_time)

    return response


# Optimized Authentication helpers with caching
@lru_cache(maxsize=1)
def get_jwt_secret() -> str:
    """Get JWT secret with caching"""
    return os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create optimized JWT access token"""
    to_encode = data.copy()
    now = datetime.utcnow()  # Use UTC for consistency

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(hours=int(os.getenv("JWT_EXPIRATION_HOURS", "1")))

    to_encode.update({
        "exp": expire,
        "iat": now,  # Issued at
        "iss": "azure-ai-copilot"  # Issuer
    })

    return jwt.encode(to_encode, get_jwt_secret(), algorithm="HS256")


async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Optimized JWT token verification with caching"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Try cached decode first
    payload = _decode_token_cached(token, get_jwt_secret())
    if payload:
        # Verify issuer
        if payload.get("iss") != "azure-ai-copilot":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer"
            )
        return payload

    # If cached decode failed, try fresh decode for better error messages
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=["HS256"])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        error_count.labels(endpoint="auth", error_type="invalid_token").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Optimized Health check endpoints
@app.get("/health")
@rate_limit(max_calls=1000, window_seconds=60)  # Higher limit for health checks
async def health_check():
    """Optimized basic health check endpoint"""
    uptime = datetime.now() - app_start_time
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",  # ISO format with UTC
        "uptime_seconds": int(uptime.total_seconds()),
        "version": "1.0.0"
    }


@app.get("/health/detailed")
async def detailed_health_check(current_user: dict = Depends(verify_token)):
    """Optimized detailed health check with comprehensive system status"""
    start_time = time.time()
    uptime = datetime.now() - app_start_time

    # Check orchestrator status
    orchestrator_status = {
        "status": "available" if orchestrator else "not_available",
        "type": "AzureAIOrchestrator" if orchestrator else None
    }

    if orchestrator:
        try:
            orchestrator_status.update(orchestrator.get_status())
        except Exception as e:
            orchestrator_status["status"] = "error"
            orchestrator_status["error"] = str(e)

    # Check Redis status with timeout
    redis_status = await redis_manager.safe_operation(
        "health_ping",
        lambda client: {"status": "connected", "ping": "pong"},
        {"status": "disconnected"}
    )

    # System metrics
    system_info = {
        "active_websockets": len(active_websockets),
        "uptime_seconds": int(uptime.total_seconds()),
        "response_time_ms": round((time.time() - start_time) * 1000, 2)
    }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "orchestrator": orchestrator_status,
        "redis": redis_status,
        "system": system_info,
        "version": "1.0.0"
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest())


# Authentication endpoints
@app.post("/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest):
    """Authenticate and receive access token"""
    # Get credentials from environment variables
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password_hash = os.getenv("ADMIN_PASSWORD_HASH")

    # If no hash is set, create one from the plain password (for development only)
    if not admin_password_hash:
        default_password = os.getenv("ADMIN_PASSWORD", "change-me-in-production")
        admin_password_hash = pwd_context.hash(default_password)
        logger.warning("Using unhashed password from environment. Set ADMIN_PASSWORD_HASH for production!")

    # Verify credentials
    if (auth_request.username == admin_username and
        pwd_context.verify(auth_request.password, admin_password_hash)):

        access_token = create_access_token(
            data={"sub": auth_request.username, "role": "owner"}
        )
        return AuthResponse(access_token=access_token)

    # Add a small delay to prevent timing attacks
    import time
    time.sleep(0.1)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@app.get("/api/v1/auth/validate")
async def validate_token(current_user: dict = Depends(verify_token)):
    """Validate authentication token"""
    return {
        "valid": True,
        "user": {
            "username": current_user.get("sub"),
            "role": current_user.get("role", "reader")
        }
    }


# Command processing endpoints
@app.post("/api/v1/command", response_model=CommandResponse)
async def process_command(
    request: CommandRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_token)
):
    """Process a natural language command"""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Processing command: {request.command} (ID: {request_id})")
    request_count.labels(method="POST", endpoint="/api/v1/command", status="processing").inc()
    
    try:
        # Add user context
        context = request.context or {}
        context["user_id"] = current_user["sub"]
        context["user_role"] = current_user.get("role", "reader")
        context["auto_approve"] = request.auto_approve
        context["dry_run"] = request.dry_run
        
        # Process command
        with command_processing_time.time():
            result = await orchestrator.process_command(request.command, context)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store in Redis cache
        await safe_redis_operation(
            "store_command_result",
            lambda: redis_client.setex(
                f"command:{request_id}",
                3600,
                json.dumps({
                    "request": request.dict(),
                    "result": result,
                    "execution_time": execution_time,
                    "user": current_user["sub"],
                    "timestamp": start_time.isoformat()
                })
            )
        )
        
        # Broadcast to WebSocket clients
        await broadcast_to_websockets({
            "type": "command_completed",
            "data": {
                "request_id": request_id,
                "command": request.command,
                "status": result.get("status", "unknown"),
                "execution_time": execution_time
            }
        })
        
        request_count.labels(method="POST", endpoint="/api/v1/command", status="success").inc()
        
        return CommandResponse(
            request_id=request_id,
            status=result.get("status", "completed"),
            result=result,
            execution_time=execution_time,
            timestamp=start_time.isoformat()
        )
    
    except Exception as e:
        logger.error(f"Command processing failed: {str(e)}")
        request_count.labels(method="POST", endpoint="/api/v1/command", status="error").inc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Command processing failed: {str(e)}"
        )


@app.get("/api/v1/command/{request_id}")
async def get_command_result(
    request_id: str,
    current_user: dict = Depends(verify_token)
):
    """Get result of a previously executed command"""
    result = await safe_redis_operation(
        "get_command_result",
        lambda: redis_client.get(f"command:{request_id}"),
        default_return=None
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command result not found"
        )

    return json.loads(result)


# Resource management endpoints
@app.post("/api/v1/resources/query")
async def query_resources(
    query: ResourceQuery,
    current_user: dict = Depends(verify_token)
):
    """Query Azure resources"""
    try:
        command = f"List all {query.resource_type or 'resources'}"
        if query.resource_group:
            command += f" in resource group {query.resource_group}"
        if query.tags:
            command += f" with tags {json.dumps(query.tags)}"

        context = {
            "user_id": current_user["sub"],
            "user_role": current_user.get("role", "reader"),
            "include_metrics": query.include_metrics
        }

        result = await orchestrator.process_command(command, context)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Resource query failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource query failed: {str(e)}"
        )


@app.post("/api/v1/resources/create")
async def create_resource(
    resource_type: str,
    specifications: Dict[str, Any],
    current_user: dict = Depends(verify_token)
):
    """Create a new Azure resource"""
    try:
        if current_user.get("role") not in ["contributor", "owner"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create resources"
            )

        # Validate resource type
        valid_types = ["vm", "storage", "network", "database", "webapp"]
        if resource_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid resource type. Must be one of: {', '.join(valid_types)}"
            )

        command = f"Create a {resource_type} with specifications: {json.dumps(specifications)}"

        context = {
            "user_id": current_user["sub"],
            "user_role": current_user.get("role"),
            "auto_approve": False
        }

        result = await orchestrator.process_command(command, context)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Resource creation failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource creation failed: {str(e)}"
        )


@app.delete("/api/v1/resources/{resource_id}")
async def delete_resource(
    resource_id: str,
    force: bool = False,
    current_user: dict = Depends(verify_token)
):
    """Delete an Azure resource"""
    try:
        if current_user.get("role") != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only owners can delete resources"
            )

        # Validate resource ID format
        if not resource_id or len(resource_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid resource ID format"
            )

        command = f"Delete resource with ID {resource_id}"
        if force:
            command += " forcefully"

        context = {
            "user_id": current_user["sub"],
            "user_role": current_user.get("role"),
            "force": force
        }

        result = await orchestrator.process_command(command, context)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Resource deletion failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource deletion failed: {str(e)}"
        )


@app.get("/api/v1/resources/{resource_id}")
async def get_resource_details(
    resource_id: str,
    current_user: dict = Depends(verify_token)
):
    """Get details of a specific Azure resource"""
    # Mock resource details
    return {
        "id": resource_id,
        "name": f"resource-{resource_id[-8:]}",
        "type": "Microsoft.Compute/virtualMachines",
        "location": "East US",
        "resourceGroup": "rg-production",
        "status": "Running",
        "properties": {
            "vmSize": "Standard_D2s_v3",
            "osType": "Linux",
            "publicIPAddress": "20.123.45.67",
            "privateIPAddress": "10.0.1.4"
        },
        "tags": {
            "Environment": "Production",
            "Team": "Backend",
            "Project": "WebApp"
        },
        "created": "2023-10-01T12:00:00Z",
        "lastModified": "2023-10-15T09:30:00Z",
        "metrics": {
            "cpuUtilization": 45.2,
            "memoryUtilization": 62.8,
            "diskUtilization": 78.5,
            "networkIn": 1024.5,
            "networkOut": 2048.3
        }
    }


@app.get("/api/v1/resources/distribution")
async def get_resource_distribution(
    current_user: dict = Depends(verify_token)
):
    """Get distribution of resources by type and location"""
    return {
        "by_type": {
            "Virtual Machines": 12,
            "Storage Accounts": 8,
            "App Services": 5,
            "SQL Databases": 3,
            "Key Vaults": 2,
            "Virtual Networks": 4
        },
        "by_location": {
            "East US": 18,
            "West US": 8,
            "Central US": 5,
            "Europe West": 3
        },
        "by_resource_group": {
            "rg-production": 15,
            "rg-staging": 8,
            "rg-development": 6,
            "rg-shared": 5
        },
        "by_status": {
            "Running": 28,
            "Stopped": 4,
            "Deallocated": 2
        },
        "total_resources": 34,
        "total_cost_monthly": 5420.50
    }


@app.get("/api/v1/metrics/summary")
async def get_metrics_summary(
    current_user: dict = Depends(verify_token)
):
    """Get summary of key metrics"""
    return {
        "infrastructure": {
            "total_resources": 34,
            "healthy_resources": 31,
            "warning_resources": 2,
            "critical_resources": 1
        },
        "performance": {
            "avg_cpu_utilization": 45.2,
            "avg_memory_utilization": 62.8,
            "avg_response_time": 125.5,
            "uptime_percentage": 99.8
        },
        "cost": {
            "current_month": 4250.75,
            "previous_month": 3980.50,
            "change_percent": 6.8,
            "projected_month": 5420.50
        },
        "security": {
            "compliance_score": 85.5,
            "critical_alerts": 2,
            "total_vulnerabilities": 5,
            "patched_systems": 28
        },
        "incidents": {
            "open_incidents": 3,
            "resolved_today": 7,
            "avg_resolution_time": "45 minutes",
            "sla_compliance": 94.2
        }
    }


# Incident management endpoints
@app.post("/api/v1/incidents/report")
async def report_incident(
    incident: IncidentReport,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_token)
):
    """Report and diagnose an incident"""
    command = f"""Diagnose incident: {incident.description}
    Severity: {incident.severity}
    Affected resources: {', '.join(incident.affected_resources)}
    Symptoms: {', '.join(incident.symptoms)}
    """
    
    context = {
        "user_id": current_user["sub"],
        "user_role": current_user.get("role"),
        "auto_remediate": incident.auto_remediate,
        "incident_data": incident.dict()
    }
    
    # Process in background for long-running diagnostics
    if incident.severity in ["high", "critical"]:
        result = await orchestrator.process_command(command, context)
    else:
        background_tasks.add_task(
            process_incident_background,
            command,
            context,
            incident.dict()
        )
        result = {
            "status": "processing",
            "message": "Incident is being processed in the background"
        }
    
    return result


async def process_incident_background(
    command: str,
    context: Dict,
    incident_data: Dict
):
    """Process incident in background"""
    result = await orchestrator.process_command(command, context)
    
    # Store result in Redis
    await safe_redis_operation(
        "store_incident_result",
        lambda: redis_client.setex(
            f"incident:{incident_data.get('description', 'unknown')[:50]}",
            86400,  # 24 hours
            json.dumps(result)
        )
    )
    
    # Notify via WebSocket
    await broadcast_to_websockets({
        "type": "incident_resolved",
        "data": {
            "incident": incident_data,
            "result": result
        }
    })


@app.get("/api/v1/incidents/active")
async def get_active_incidents(
    current_user: dict = Depends(verify_token)
):
    """Get list of active incidents"""
    # Get all incident keys from Redis
    incident_keys = await safe_redis_operation(
        "get_incident_keys",
        lambda: redis_client.keys("incident:*"),
        default_return=[]
    )

    incidents = []
    for key in incident_keys:
        incident_data = await safe_redis_operation(
            f"get_incident_{key}",
            lambda: redis_client.get(key),
            default_return=None
        )
        if incident_data:
            incidents.append(json.loads(incident_data))

    return {"incidents": incidents, "count": len(incidents)}


@app.get("/api/v1/incidents/recent")
async def get_recent_incidents(
    days: int = 7,
    current_user: dict = Depends(verify_token)
):
    """Get recent incidents from the past N days"""
    # Get incidents from Redis and filter by date
    incident_keys = await safe_redis_operation(
        "get_recent_incident_keys",
        lambda: redis_client.keys("incident:*"),
        default_return=[]
    )

    recent_incidents = []
    cutoff_date = datetime.now() - timedelta(days=days)

    for key in incident_keys:
        incident_data = await safe_redis_operation(
            f"get_recent_incident_{key}",
            lambda: redis_client.get(key),
            default_return=None
        )
        if incident_data:
            incident = json.loads(incident_data)
            # Mock date check (in real scenario would parse actual date)
            recent_incidents.append(incident)

    return {"incidents": recent_incidents[:10], "days": days}  # Limit to 10 recent


# Cost optimization endpoints
@app.post("/api/v1/cost/analyze")
async def analyze_costs(
    request: CostAnalysisRequest,
    current_user: dict = Depends(verify_token)
):
    """Analyze costs and find optimization opportunities"""
    command = f"""Analyze Azure costs for {request.scope}
    Timeframe: {request.timeframe}
    Optimization level: {request.optimization_level}
    """
    
    if request.resource_group:
        command += f"Resource group: {request.resource_group}"
    
    context = {
        "user_id": current_user["sub"],
        "user_role": current_user.get("role"),
        "cost_analysis": request.dict()
    }
    
    result = await orchestrator.process_command(command, context)
    return result


@app.post("/api/v1/cost/optimize")
async def optimize_costs(
    optimization_ids: List[str],
    auto_apply: bool = False,
    current_user: dict = Depends(verify_token)
):
    """Apply cost optimization recommendations"""
    if current_user.get("role") not in ["contributor", "owner"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to apply optimizations"
        )
    
    command = f"Apply cost optimizations: {', '.join(optimization_ids)}"
    
    context = {
        "user_id": current_user["sub"],
        "user_role": current_user.get("role"),
        "auto_apply": auto_apply,
        "optimization_ids": optimization_ids
    }
    
    result = await orchestrator.process_command(command, context)
    return result


@app.get("/api/v1/cost/trend")
async def get_cost_trend(
    timeframe: str = "30d",
    current_user: dict = Depends(verify_token)
):
    """Get cost trend data over time"""
    # Mock cost trend data
    import random
    from datetime import datetime, timedelta

    days = int(timeframe.replace('d', ''))
    dates = []
    costs = []

    for i in range(days):
        date = datetime.now() - timedelta(days=days-i-1)
        dates.append(date.strftime('%Y-%m-%d'))
        # Generate mock trending cost data
        base_cost = 1000 + (i * 20) + random.randint(-100, 100)
        costs.append(max(0, base_cost))

    return {
        "timeframe": timeframe,
        "data": [
            {"date": date, "cost": cost}
            for date, cost in zip(dates, costs)
        ],
        "total_cost": sum(costs),
        "average_daily_cost": sum(costs) / len(costs),
        "trend": "increasing" if costs[-1] > costs[0] else "decreasing"
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "data": {"status": "connected", "timestamp": datetime.now().isoformat()}
        })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "subscribe":
                # Handle subscription to specific events
                pass
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_to_websockets(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# History and analytics endpoints
@app.get("/api/v1/history")
async def get_command_history(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(verify_token)
):
    """Get command history for the current user"""
    # Get all command keys from Redis
    command_keys = await safe_redis_operation(
        "get_command_keys",
        lambda: redis_client.keys("command:*"),
        default_return=[]
    )

    history = []
    for key in command_keys[offset:offset + limit]:
        command_data = await safe_redis_operation(
            f"get_command_{key}",
            lambda: redis_client.get(key),
            default_return=None
        )
        if command_data:
            data = json.loads(command_data)
            if data.get("user") == current_user["sub"] or current_user.get("role") == "owner":
                history.append(data)
    
    # Sort by timestamp
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {
        "history": history,
        "total": len(command_keys),
        "limit": limit,
        "offset": offset
    }


# Compliance endpoints
@app.post("/api/v1/compliance/check")
async def check_compliance(
    frameworks: List[str],
    current_user: dict = Depends(verify_token)
):
    """Run compliance checks against specified frameworks"""
    command = f"Run compliance checks for frameworks: {', '.join(frameworks)}"

    context = {
        "user_id": current_user["sub"],
        "user_role": current_user.get("role"),
        "frameworks": frameworks
    }

    result = await orchestrator.process_command(command, context)
    return result


@app.get("/api/v1/compliance/report")
async def get_compliance_report(
    framework: str = "CIS",
    current_user: dict = Depends(verify_token)
):
    """Get latest compliance report"""
    # Mock compliance report data
    return {
        "framework": framework,
        "generated_at": datetime.now().isoformat(),
        "overall_score": 85.5,
        "total_checks": 42,
        "passed": 36,
        "failed": 6,
        "categories": {
            "Identity and Access": {"score": 90, "checks": 12, "passed": 11},
            "Network Security": {"score": 80, "checks": 15, "passed": 12},
            "Data Protection": {"score": 88, "checks": 10, "passed": 9},
            "Logging and Monitoring": {"score": 75, "checks": 5, "passed": 4}
        },
        "critical_findings": [
            {
                "id": "CIS-2.1",
                "title": "Security Center standard tier not enabled",
                "severity": "High",
                "status": "Failed"
            },
            {
                "id": "CIS-3.1",
                "title": "Storage secure transfer not enforced",
                "severity": "High",
                "status": "Failed"
            }
        ]
    }


# Predictions endpoints
@app.get("/api/v1/predictions")
async def get_predictions(
    time_horizon: str = "7d",
    current_user: dict = Depends(verify_token)
):
    """Get resource and cost predictions"""
    # Mock prediction data
    days = int(time_horizon.replace('d', ''))

    return {
        "time_horizon": time_horizon,
        "generated_at": datetime.now().isoformat(),
        "cost_prediction": {
            "current_monthly": 5420.50,
            "predicted_monthly": 6200.75,
            "change_percent": 14.4,
            "trend": "increasing"
        },
        "resource_predictions": {
            "cpu_utilization": {
                "current_avg": 45.2,
                "predicted_avg": 52.8,
                "trend": "increasing"
            },
            "memory_utilization": {
                "current_avg": 62.1,
                "predicted_avg": 68.9,
                "trend": "increasing"
            },
            "storage_growth": {
                "current_gb": 1250,
                "predicted_gb": 1390,
                "growth_rate": "11.2%"
            }
        },
        "recommendations": [
            "Consider scaling up VM instances due to predicted CPU increase",
            "Review storage retention policies to manage growth",
            "Evaluate auto-scaling policies for cost optimization"
        ]
    }


@app.get("/api/v1/predictions/summary")
async def get_predictions_summary(
    current_user: dict = Depends(verify_token)
):
    """Get predictions summary dashboard"""
    return {
        "summary": {
            "cost_trend": "increasing",
            "resource_utilization": "moderate",
            "capacity_alerts": 2,
            "optimization_opportunities": 5
        },
        "alerts": [
            {
                "type": "cost",
                "message": "Monthly cost predicted to exceed budget by 15%",
                "severity": "high"
            },
            {
                "type": "capacity",
                "message": "Storage projected to reach 90% capacity in 14 days",
                "severity": "medium"
            }
        ],
        "next_review": (datetime.now() + timedelta(days=7)).isoformat()
    }


@app.get("/api/v1/analytics/usage")
async def get_usage_analytics(
    timeframe: str = "7d",
    current_user: dict = Depends(verify_token)
):
    """Get usage analytics"""
    if current_user.get("role") != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners can view analytics"
        )
    
    # This would aggregate data from Redis/database
    return {
        "timeframe": timeframe,
        "total_commands": 1247,
        "unique_users": 23,
        "most_used_intents": [
            {"intent": "resource_query", "count": 456},
            {"intent": "incident_diagnosis", "count": 312},
            {"intent": "cost_optimization", "count": 234}
        ],
        "average_execution_time": 2.3,
        "success_rate": 0.94
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
