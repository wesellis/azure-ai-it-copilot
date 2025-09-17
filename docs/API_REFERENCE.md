# 📚 API Reference

## Base URL
```
Production: https://api.azure-ai-copilot.com/v1
Development: http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication using Bearer tokens:

```http
Authorization: Bearer <your-jwt-token>
```

### Get Access Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "admin@company.com",
  "password": "your-password",
  "grant_type": "password"
}
```

#### Response
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh-token-here",
  "scope": "read write execute"
}
```

## Core Endpoints

### 1. Execute Command

Execute a natural language command on your infrastructure.

```http
POST /api/v1/command
Authorization: Bearer <token>
Content-Type: application/json

{
  "command": "Create a Linux VM with 8GB RAM in East US",
  "context": {
    "resource_group": "rg-production",
    "subscription_id": "optional-subscription-id"
  },
  "options": {
    "dry_run": false,
    "require_approval": true,
    "timeout_seconds": 300
  }
}
```

#### Response
```json
{
  "request_id": "req_abc123",
  "status": "pending_approval",
  "interpretation": {
    "intent": "resource_creation",
    "confidence": 0.95,
    "entities": {
      "resource_type": "virtual_machine",
      "os": "linux",
      "memory": "8GB",
      "location": "eastus"
    }
  },
  "execution_plan": {
    "steps": [
      {
        "order": 1,
        "action": "create_resource_group",
        "resource": "rg-vm-prod",
        "status": "pending"
      },
      {
        "order": 2,
        "action": "create_virtual_machine",
        "resource": "vm-linux-001",
        "properties": {
          "size": "Standard_D2s_v3",
          "os": "Ubuntu 22.04",
          "memory": "8GB"
        },
        "status": "pending"
      }
    ],
    "estimated_cost": "$150/month",
    "estimated_time": "5 minutes"
  },
  "approval_url": "/api/v1/approvals/req_abc123"
}
```

### 2. List Commands

Get history of executed commands.

```http
GET /api/v1/commands?limit=50&status=completed&from=2025-01-01
Authorization: Bearer <token>
```

#### Response
```json
{
  "commands": [
    {
      "id": "cmd_123",
      "command": "List all VMs",
      "executed_at": "2025-01-15T10:30:00Z",
      "executed_by": "admin@company.com",
      "status": "completed",
      "execution_time_ms": 1250,
      "resources_affected": 0
    }
  ],
  "pagination": {
    "total": 150,
    "page": 1,
    "per_page": 50
  }
}
```

### 3. Get Command Status

```http
GET /api/v1/commands/{command_id}
Authorization: Bearer <token>
```

#### Response
```json
{
  "id": "cmd_123",
  "status": "completed",
  "progress": 100,
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:05Z",
  "results": {
    "resources_created": [
      {
        "type": "Microsoft.Compute/virtualMachines",
        "name": "vm-linux-001",
        "id": "/subscriptions/.../resourceGroups/.../providers/..."
      }
    ],
    "output": "Successfully created Linux VM with 8GB RAM in East US"
  }
}
```

## Agent APIs

### Infrastructure Agent

#### List Resources
```http
POST /api/v1/agents/infrastructure/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "Show all VMs with tag environment=production",
  "filters": {
    "resource_type": "Microsoft.Compute/virtualMachines",
    "tags": {
      "environment": "production"
    }
  }
}
```

#### Create Resource
```http
POST /api/v1/agents/infrastructure/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "resource_type": "storage_account",
  "properties": {
    "name": "stgprod001",
    "sku": "Standard_LRS",
    "location": "eastus2",
    "kind": "StorageV2"
  }
}
```

### Incident Response Agent

#### Diagnose Issue
```http
POST /api/v1/agents/incident/diagnose
Authorization: Bearer <token>
Content-Type: application/json

{
  "resource_id": "/subscriptions/.../vm-prod-001",
  "symptoms": ["high_cpu", "slow_response"],
  "timeframe": "last_hour"
}
```

#### Response
```json
{
  "diagnosis": {
    "root_cause": "Memory leak in application process",
    "confidence": 0.89,
    "evidence": [
      "Memory usage increased from 45% to 95% over 2 hours",
      "Process 'app.exe' consuming 8GB RAM",
      "No recent deployments or changes"
    ]
  },
  "recommended_actions": [
    {
      "action": "restart_process",
      "description": "Restart the application process",
      "risk_level": "low",
      "estimated_fix_time": "2 minutes"
    },
    {
      "action": "scale_up",
      "description": "Increase VM size to provide more memory",
      "risk_level": "medium",
      "estimated_fix_time": "10 minutes"
    }
  ]
}
```

#### Auto-Remediate
```http
POST /api/v1/agents/incident/remediate
Authorization: Bearer <token>
Content-Type: application/json

{
  "diagnosis_id": "diag_456",
  "action": "restart_process",
  "auto_rollback": true
}
```

### Cost Optimization Agent

#### Get Recommendations
```http
GET /api/v1/agents/cost/recommendations
Authorization: Bearer <token>
```

#### Response
```json
{
  "total_potential_savings": "$5,240/month",
  "recommendations": [
    {
      "type": "unused_resources",
      "description": "15 VMs stopped for >30 days",
      "savings": "$2,100/month",
      "action": "delete_resources",
      "resources": ["vm-dev-001", "vm-test-002", "..."]
    },
    {
      "type": "rightsizing",
      "description": "8 over-provisioned VMs",
      "savings": "$1,540/month",
      "action": "resize_vms",
      "details": [
        {
          "vm": "vm-prod-003",
          "current_size": "D8s_v3",
          "recommended_size": "D4s_v3",
          "savings": "$192/month"
        }
      ]
    },
    {
      "type": "reserved_instances",
      "description": "Buy RIs for steady-state workloads",
      "savings": "$1,600/month",
      "action": "purchase_reserved_instances",
      "recommendations": [
        {
          "vm_size": "D4s_v3",
          "quantity": 5,
          "term": "3-year",
          "savings": "$320/month per instance"
        }
      ]
    }
  ]
}
```

#### Implement Optimization
```http
POST /api/v1/agents/cost/optimize
Authorization: Bearer <token>
Content-Type: application/json

{
  "recommendation_id": "rec_789",
  "actions": ["delete_unused", "rightsize_vms"],
  "approval": "approved",
  "schedule": "immediate"
}
```

### Compliance Agent

#### Run Compliance Check
```http
POST /api/v1/agents/compliance/scan
Authorization: Bearer <token>
Content-Type: application/json

{
  "standards": ["SOC2", "HIPAA", "CIS"],
  "scope": "all_resources"
}
```

#### Response
```json
{
  "scan_id": "scan_101",
  "compliance_score": 87.5,
  "standards": {
    "SOC2": {
      "score": 92,
      "passed": 46,
      "failed": 4,
      "total": 50
    },
    "HIPAA": {
      "score": 85,
      "passed": 34,
      "failed": 6,
      "total": 40
    },
    "CIS": {
      "score": 86,
      "passed": 172,
      "failed": 28,
      "total": 200
    }
  },
  "critical_findings": [
    {
      "standard": "HIPAA",
      "rule": "Encryption at rest",
      "resources": ["storage-account-001"],
      "remediation": "Enable storage service encryption"
    }
  ]
}
```

### Predictive Agent

#### Get Predictions
```http
GET /api/v1/agents/predictive/forecast
Authorization: Bearer <token>
```

#### Response
```json
{
  "predictions": [
    {
      "type": "capacity",
      "resource": "vm-prod-001",
      "metric": "cpu_utilization",
      "prediction": "Will exceed 90% in 3 days",
      "confidence": 0.92,
      "recommendation": "Scale up to D8s_v3"
    },
    {
      "type": "failure",
      "resource": "disk-data-002",
      "prediction": "Likely failure in 7-10 days",
      "confidence": 0.78,
      "indicators": ["increasing error rate", "SMART warnings"],
      "recommendation": "Migrate data to new disk"
    },
    {
      "type": "cost",
      "prediction": "Monthly cost will increase by 25% next month",
      "confidence": 0.85,
      "drivers": ["increased storage usage", "new VM deployments"],
      "recommendation": "Review and optimize before month-end"
    }
  ]
}
```

## WebSocket API

### Real-time Command Execution

```javascript
const ws = new WebSocket('wss://api.azure-ai-copilot.com/v1/ws');

ws.on('open', () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));

  // Execute command
  ws.send(JSON.stringify({
    type: 'command',
    data: {
      command: 'Monitor all production resources',
      stream: true
    }
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);

  switch(message.type) {
    case 'progress':
      console.log(`Progress: ${message.percentage}%`);
      break;
    case 'log':
      console.log(`Log: ${message.content}`);
      break;
    case 'result':
      console.log(`Result: ${message.data}`);
      break;
    case 'error':
      console.error(`Error: ${message.error}`);
      break;
  }
});
```

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The specified resource was not found",
    "details": {
      "resource_id": "/subscriptions/.../vm-missing",
      "suggestion": "Check if the resource exists or has been deleted"
    },
    "request_id": "req_xyz789",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource doesn't exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INVALID_REQUEST` | 400 | Malformed request |
| `COMMAND_FAILED` | 500 | Command execution failed |
| `AI_SERVICE_ERROR` | 503 | AI service unavailable |

## Rate Limiting

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642252800
```

- **Default Limits**: 1000 requests/hour
- **Burst Limit**: 100 requests/minute
- **Command Execution**: 100 commands/hour

## Pagination

```http
GET /api/v1/resources?page=2&per_page=50
```

#### Response Headers
```http
X-Total-Count: 250
X-Page: 2
X-Per-Page: 50
X-Total-Pages: 5
Link: <.../resources?page=3>; rel="next", <.../resources?page=1>; rel="prev"
```

## SDKs

### Python
```python
from azure_ai_copilot import Client

client = Client(
    api_key="your-api-key",
    endpoint="https://api.azure-ai-copilot.com"
)

# Execute command
result = client.execute_command(
    "Create a VM in East US",
    dry_run=False
)

print(result.status)
print(result.output)
```

### JavaScript/TypeScript
```typescript
import { AzureAICopilot } from '@azure-ai-copilot/sdk';

const client = new AzureAICopilot({
  apiKey: 'your-api-key'
});

// Execute command
const result = await client.executeCommand({
  command: 'Create a VM in East US',
  options: { dryRun: false }
});

console.log(result.status);
console.log(result.output);
```

### PowerShell
```powershell
# Install module
Install-Module -Name AzureAICopilot

# Connect
Connect-AICopilot -ApiKey "your-api-key"

# Execute command
Invoke-AICopilotCommand -Command "Create a VM in East US"
```

## Webhooks

Register webhooks to receive notifications:

```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-domain.com/webhook",
  "events": ["command.completed", "incident.detected", "cost.threshold"],
  "secret": "webhook-secret"
}
```

### Webhook Payload
```json
{
  "event": "command.completed",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "command_id": "cmd_123",
    "status": "success",
    "command": "Create VM",
    "executed_by": "admin@company.com"
  },
  "signature": "sha256=..."
}
```

## API Versioning

The API uses URL versioning. Current version: `v1`

Future versions will maintain backward compatibility for 12 months after deprecation notice.

## Support

- 📖 [OpenAPI Specification](https://api.azure-ai-copilot.com/openapi.json)
- 🔧 [Postman Collection](https://www.postman.com/azure-ai-copilot)
- 💬 [API Support Forum](https://forum.azure-ai-copilot.com)
- 🐛 [Report API Issues](https://github.com/yourusername/azure-ai-it-copilot/issues)