"""
Microsoft Graph API Integration
Handles user management, Teams, and Office 365 integration
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import msal
import aiohttp
from azure.identity import ClientSecretCredential

logger = logging.getLogger(__name__)


class GraphConnector:
    """Microsoft Graph API connector for M365 integration"""

    def __init__(self):
        """Initialize Graph connector with Azure AD authentication"""
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")

        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.graph_url = "https://graph.microsoft.com/v1.0"

        # Initialize MSAL client
        self.app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret
        )

        self._token = None
        self._token_expiry = None

    async def get_access_token(self) -> str:
        """Get or refresh access token"""
        if self._token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return self._token

        result = self.app.acquire_token_silent(self.scope, account=None)

        if not result:
            result = self.app.acquire_token_for_client(scopes=self.scope)

        if "access_token" in result:
            self._token = result["access_token"]
            # Token usually expires in 1 hour, refresh 5 minutes early
            self._token_expiry = datetime.utcnow() + timedelta(minutes=55)
            return self._token

        raise Exception(f"Failed to acquire token: {result.get('error_description')}")

    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Graph API"""
        token = await self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        url = f"{self.graph_url}{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                json=data,
                params=params,
                headers=headers
            ) as response:
                if response.status == 204:
                    return {"status": "success"}

                result = await response.json()

                if response.status >= 400:
                    logger.error(f"Graph API error: {result}")
                    raise Exception(f"Graph API error: {result.get('error', {}).get('message')}")

                return result

    # User Management
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user details by ID or UPN"""
        return await self.make_request("GET", f"/users/{user_id}")

    async def list_users(
        self,
        filter_query: Optional[str] = None,
        select: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List users with optional filtering"""
        params = {}

        if filter_query:
            params["$filter"] = filter_query

        if select:
            params["$select"] = ",".join(select)

        result = await self.make_request("GET", "/users", params=params)
        return result.get("value", [])

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user"""
        required_fields = ["accountEnabled", "displayName", "mailNickname",
                          "userPrincipalName", "passwordProfile"]

        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"Missing required field: {field}")

        return await self.make_request("POST", "/users", data=user_data)

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user properties"""
        return await self.make_request("PATCH", f"/users/{user_id}", data=updates)

    async def delete_user(self, user_id: str) -> Dict[str, Any]:
        """Delete a user"""
        return await self.make_request("DELETE", f"/users/{user_id}")

    async def reset_password(self, user_id: str, new_password: str) -> Dict[str, Any]:
        """Reset user password"""
        data = {
            "passwordProfile": {
                "password": new_password,
                "forceChangePasswordNextSignIn": True
            }
        }
        return await self.make_request("PATCH", f"/users/{user_id}", data=data)

    # Group Management
    async def list_groups(self) -> List[Dict[str, Any]]:
        """List all groups"""
        result = await self.make_request("GET", "/groups")
        return result.get("value", [])

    async def get_group_members(self, group_id: str) -> List[Dict[str, Any]]:
        """Get members of a group"""
        result = await self.make_request("GET", f"/groups/{group_id}/members")
        return result.get("value", [])

    async def add_group_member(self, group_id: str, user_id: str) -> Dict[str, Any]:
        """Add user to group"""
        data = {
            "@odata.id": f"{self.graph_url}/users/{user_id}"
        }
        return await self.make_request("POST", f"/groups/{group_id}/members/$ref", data=data)

    async def remove_group_member(self, group_id: str, user_id: str) -> Dict[str, Any]:
        """Remove user from group"""
        return await self.make_request("DELETE", f"/groups/{group_id}/members/{user_id}/$ref")

    # Teams Operations
    async def list_teams(self, user_id: str) -> List[Dict[str, Any]]:
        """List teams for a user"""
        result = await self.make_request("GET", f"/users/{user_id}/joinedTeams")
        return result.get("value", [])

    async def create_team(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new team"""
        return await self.make_request("POST", "/teams", data=team_data)

    async def send_teams_message(
        self,
        team_id: str,
        channel_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Send message to Teams channel"""
        data = {
            "body": {
                "content": message,
                "contentType": "text"
            }
        }
        return await self.make_request(
            "POST",
            f"/teams/{team_id}/channels/{channel_id}/messages",
            data=data
        )

    # Mail Operations
    async def send_mail(
        self,
        sender_id: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        cc_recipients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Send email on behalf of a user"""

        message = {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body
            },
            "toRecipients": [
                {"emailAddress": {"address": email}} for email in to_recipients
            ]
        }

        if cc_recipients:
            message["ccRecipients"] = [
                {"emailAddress": {"address": email}} for email in cc_recipients
            ]

        data = {"message": message, "saveToSentItems": "true"}

        return await self.make_request(
            "POST",
            f"/users/{sender_id}/sendMail",
            data=data
        )

    # Calendar Operations
    async def get_calendar_events(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get calendar events for a user"""
        params = {
            "$filter": f"start/dateTime ge '{start_date.isoformat()}' and end/dateTime le '{end_date.isoformat()}'",
            "$orderby": "start/dateTime"
        }

        result = await self.make_request(
            "GET",
            f"/users/{user_id}/events",
            params=params
        )

        return result.get("value", [])

    async def create_calendar_event(
        self,
        user_id: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create calendar event"""
        return await self.make_request(
            "POST",
            f"/users/{user_id}/events",
            data=event_data
        )

    # OneDrive Operations
    async def list_drive_items(
        self,
        user_id: str,
        folder_path: str = "/"
    ) -> List[Dict[str, Any]]:
        """List files in user's OneDrive"""
        endpoint = f"/users/{user_id}/drive/root/children"

        if folder_path != "/":
            endpoint = f"/users/{user_id}/drive/root:/{folder_path}:/children"

        result = await self.make_request("GET", endpoint)
        return result.get("value", [])

    async def upload_file_to_onedrive(
        self,
        user_id: str,
        file_path: str,
        content: bytes
    ) -> Dict[str, Any]:
        """Upload file to OneDrive"""
        endpoint = f"/users/{user_id}/drive/root:/{file_path}:/content"

        # For large files, would need to implement resumable upload
        if len(content) > 4 * 1024 * 1024:  # 4MB
            raise ValueError("File too large for simple upload. Use resumable upload.")

        return await self.make_request("PUT", endpoint, data=content)

    # License Management
    async def get_user_licenses(self, user_id: str) -> List[Dict[str, Any]]:
        """Get licenses assigned to user"""
        result = await self.make_request("GET", f"/users/{user_id}/licenseDetails")
        return result.get("value", [])

    async def assign_license(
        self,
        user_id: str,
        sku_id: str
    ) -> Dict[str, Any]:
        """Assign license to user"""
        data = {
            "addLicenses": [{
                "skuId": sku_id
            }],
            "removeLicenses": []
        }
        return await self.make_request(
            "POST",
            f"/users/{user_id}/assignLicense",
            data=data
        )

    # Reporting and Analytics
    async def get_usage_report(
        self,
        report_type: str,
        period: str = "D7"
    ) -> bytes:
        """Get usage reports (requires Reports.Read.All permission)"""
        endpoint = f"/reports/get{report_type}(period='{period}')"

        # Reports return CSV data, not JSON
        token = await self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.graph_url}{endpoint}",
                headers=headers
            ) as response:
                return await response.read()

    # Intune/Device Management
    async def list_managed_devices(self) -> List[Dict[str, Any]]:
        """List Intune managed devices"""
        result = await self.make_request("GET", "/deviceManagement/managedDevices")
        return result.get("value", [])

    async def get_device_compliance(self, device_id: str) -> Dict[str, Any]:
        """Get device compliance status"""
        return await self.make_request(
            "GET",
            f"/deviceManagement/managedDevices/{device_id}/deviceCompliancePolicyStates"
        )

    async def wipe_device(self, device_id: str) -> Dict[str, Any]:
        """Remote wipe a device"""
        return await self.make_request(
            "POST",
            f"/deviceManagement/managedDevices/{device_id}/wipe"
        )


# Example usage
async def main():
    """Example usage of Graph connector"""
    connector = GraphConnector()

    # List users
    users = await connector.list_users(
        select=["displayName", "mail", "userPrincipalName"]
    )

    print(f"Found {len(users)} users")

    for user in users[:5]:
        print(f"- {user.get('displayName')} ({user.get('mail')})")


if __name__ == "__main__":
    asyncio.run(main())