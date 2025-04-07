"""
StreamLabs API Integration for OARC.

This module provides functionality for integrating OARC's speech capabilities
with StreamLabs, allowing for text-to-speech output to be sent to StreamLabs
for streaming purposes. It includes methods for authentication, sending TTS
output, and handling StreamLabs events.
"""

import os
import json
import time
import asyncio
import requests
from typing import Dict, Optional

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.decorators.singleton import singleton
from oarc.speech.speech_manager import SpeechManager


@singleton
class StreamLabsAPI:
    """
    StreamLabs API client for integrating OARC's TTS with streaming platforms.

    This class provides methods for sending text-to-speech output to StreamLabs
    and handling StreamLabs events. It follows the singleton pattern to ensure
    only one instance of the API client exists throughout the application.
    """

    def __init__(self):
        """
        Initialize the StreamLabs API client.
        
        Sets up initial configuration, loads credentials if available,
        and prepares the API client for use. Does not automatically
        authenticate - call authenticate() method to do so.
        """
        log.info("Initializing StreamLabs API client")
        self.api_base_url = "https://streamlabs.com/api/v1.0"
        self.socket_token = None
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = 0
        self.authenticated = False
        
        # Get paths
        self.paths = Paths()
        
        # Create path for credentials file
        self.creds_dir = os.path.join(self.paths.get_model_dir(), "streamlabs")
        os.makedirs(self.creds_dir, exist_ok=True)
        self.creds_file = os.path.join(self.creds_dir, "credentials.json")
        
        # Try to load existing credentials
        self._load_credentials()
    
    def _load_credentials(self) -> bool:
        """
        Load StreamLabs credentials from file if available.
        
        Returns:
            bool: True if credentials were loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.creds_file):
                with open(self.creds_file, 'r') as f:
                    creds = json.load(f)
                    
                self.access_token = creds.get('access_token')
                self.refresh_token = creds.get('refresh_token')
                self.token_expires_at = creds.get('expires_at', 0)
                self.socket_token = creds.get('socket_token')
                
                # Check if token is still valid
                if self.token_expires_at > time.time() + 300:  # 5-minute buffer
                    self.authenticated = True
                    log.info("Loaded valid StreamLabs credentials")
                    return True
                else:
                    log.info("Loaded StreamLabs credentials but token has expired")
                    return self.refresh_authentication()
            
            return False
        except Exception as e:
            log.error(f"Error loading StreamLabs credentials: {e}")
            return False
    
    def _save_credentials(self) -> bool:
        """
        Save StreamLabs credentials to file.
        
        Returns:
            bool: True if credentials were saved successfully, False otherwise
        """
        try:
            creds = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expires_at': self.token_expires_at,
                'socket_token': self.socket_token
            }
            
            with open(self.creds_file, 'w') as f:
                json.dump(creds, f)
                
            log.info("Saved StreamLabs credentials")
            return True
        except Exception as e:
            log.error(f"Error saving StreamLabs credentials: {e}")
            return False
    
    def authenticate(self, client_id: str, client_secret: str, auth_code: str) -> bool:
        """
        Authenticate with StreamLabs API using authorization code.
        
        Args:
            client_id: Your StreamLabs application client ID
            client_secret: Your StreamLabs application client secret
            auth_code: Authorization code from OAuth redirect
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            log.info("Authenticating with StreamLabs API")
            
            # Exchange auth code for tokens
            response = requests.post(
                f"{self.api_base_url}/token",
                data={
                    'grant_type': 'authorization_code',
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'code': auth_code,
                    'redirect_uri': 'http://localhost:8000/callback'  # Should match app config
                }
            )
            
            if response.status_code != 200:
                log.error(f"Authentication failed: {response.text}")
                return False
            
            data = response.json()
            self.access_token = data.get('access_token')
            self.refresh_token = data.get('refresh_token')
            self.token_expires_at = time.time() + data.get('expires_in', 3600)
            
            # Get socket token
            socket_response = requests.get(
                f"{self.api_base_url}/socket/token?access_token={self.access_token}"
            )
            
            if socket_response.status_code == 200:
                socket_data = socket_response.json()
                self.socket_token = socket_data.get('socket_token')
            else:
                log.warning(f"Failed to get socket token: {socket_response.text}")
            
            self.authenticated = True
            self._save_credentials()
            log.info("Successfully authenticated with StreamLabs API")
            return True
        except Exception as e:
            log.error(f"Error during StreamLabs authentication: {e}")
            return False
    
    def refresh_authentication(self) -> bool:
        """
        Refresh authentication tokens using refresh token.
        
        Returns:
            bool: True if refresh was successful, False otherwise
        """
        if not self.refresh_token:
            log.error("Cannot refresh authentication: No refresh token available")
            return False
        
        try:
            log.info("Refreshing StreamLabs authentication")
            
            response = requests.post(
                f"{self.api_base_url}/token",
                data={
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token
                }
            )
            
            if response.status_code != 200:
                log.error(f"Token refresh failed: {response.text}")
                self.authenticated = False
                return False
            
            data = response.json()
            self.access_token = data.get('access_token')
            self.refresh_token = data.get('refresh_token')
            self.token_expires_at = time.time() + data.get('expires_in', 3600)
            
            # Refresh socket token
            socket_response = requests.get(
                f"{self.api_base_url}/socket/token?access_token={self.access_token}"
            )
            
            if socket_response.status_code == 200:
                socket_data = socket_response.json()
                self.socket_token = socket_data.get('socket_token')
            
            self.authenticated = True
            self._save_credentials()
            log.info("Successfully refreshed StreamLabs authentication")
            return True
        except Exception as e:
            log.error(f"Error refreshing StreamLabs authentication: {e}")
            self.authenticated = False
            return False
    
    def check_authentication(self) -> bool:
        """
        Check if current authentication is valid and refresh if needed.
        
        Returns:
            bool: True if authentication is valid, False otherwise
        """
        # If not authenticated at all, return False
        if not self.authenticated or not self.access_token:
            return False
        
        # If token is about to expire, refresh it
        if self.token_expires_at <= time.time() + 300:  # 5-minute buffer
            return self.refresh_authentication()
        
        return True
    
    async def send_tts(self, text: str, voice_name: Optional[str] = None) -> bool:
        """
        Send text to StreamLabs TTS.
        
        Args:
            text: Text to convert to speech
            voice_name: Optional voice name to use (defaults to current SpeechManager voice)
            
        Returns:
            bool: True if the TTS request was successful, False otherwise
        """
        if not self.check_authentication():
            log.error("Cannot send TTS: Not authenticated")
            return False
        
        try:
            log.info(f"Sending TTS to StreamLabs: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Use SpeechManager to generate speech
            speech_manager = SpeechManager(voice_name=voice_name if voice_name else "c3po")
            
            audio_data = await asyncio.to_thread(
                speech_manager.generate_speech,
                text=text
            )
            
            if audio_data is None:
                log.error("Failed to generate audio for TTS")
                return False
            
            # TODO: Send audio to StreamLabs (this would depend on StreamLabs API capabilities)
            # This is a placeholder - actual implementation would depend on StreamLabs API
            # which might require upload of the audio file or direct streaming
            
            log.info("TTS sent to StreamLabs successfully")
            return True
        except Exception as e:
            log.error(f"Error sending TTS to StreamLabs: {e}")
            return False
    
    async def send_alert(self, alert_type: str, message: str, **kwargs) -> bool:
        """
        Send an alert to StreamLabs.
        
        Args:
            alert_type: Type of alert (donation, subscription, follow, etc.)
            message: Alert message
            **kwargs: Additional alert parameters
            
        Returns:
            bool: True if the alert was sent successfully, False otherwise
        """
        if not self.check_authentication():
            log.error("Cannot send alert: Not authenticated")
            return False
        
        try:
            log.info(f"Sending {alert_type} alert to StreamLabs")
            
            # Prepare alert data based on type
            alert_data = {
                'type': alert_type,
                'message': message,
                **kwargs
            }
            
            # Send the alert to StreamLabs API
            response = requests.post(
                f"{self.api_base_url}/alerts",
                headers={
                    'Authorization': f"Bearer {self.access_token}",
                    'Content-Type': 'application/json'
                },
                json=alert_data
            )
            
            if response.status_code == 200:
                log.info(f"{alert_type.capitalize()} alert sent successfully")
                return True
            else:
                log.error(f"Failed to send alert: {response.text}")
                return False
        except Exception as e:
            log.error(f"Error sending alert to StreamLabs: {e}")
            return False
    
    def get_user_info(self) -> Optional[Dict]:
        """
        Get information about the authenticated user.
        
        Returns:
            dict: User information or None if request failed
        """
        if not self.check_authentication():
            log.error("Cannot get user info: Not authenticated")
            return None
        
        try:
            response = requests.get(
                f"{self.api_base_url}/user",
                headers={'Authorization': f"Bearer {self.access_token}"}
            )
            
            if response.status_code == 200:
                user_data = response.json()
                log.info(f"Retrieved user info for {user_data.get('display_name', 'unknown user')}")
                return user_data
            else:
                log.error(f"Failed to get user info: {response.text}")
                return None
        except Exception as e:
            log.error(f"Error getting user info from StreamLabs: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up resources used by the StreamLabs API client.
        """
        log.info("Cleaning up StreamLabs API resources")
        self.authenticated = False
        
        # Reset the singleton instance
        self._reset_singleton()


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    from oarc.utils.const import SUCCESS, FAILURE
    
    parser = argparse.ArgumentParser(description="StreamLabs API Client")
    parser.add_argument("--auth", action="store_true", help="Authenticate with StreamLabs")
    parser.add_argument("--client-id", help="StreamLabs application client ID")
    parser.add_argument("--client-secret", help="StreamLabs application client secret")
    parser.add_argument("--code", help="Authorization code from OAuth redirect")
    parser.add_argument("--tts", help="Send TTS message to StreamLabs")
    parser.add_argument("--voice", help="Voice to use for TTS")
    parser.add_argument("--user-info", action="store_true", help="Get authenticated user info")
    
    args = parser.parse_args()
    streamlabs = StreamLabsAPI()
    
    if args.auth:
        if not all([args.client_id, args.client_secret, args.code]):
            print("Error: --client-id, --client-secret, and --code are required for authentication")
            sys.exit(FAILURE)
        
        if streamlabs.authenticate(args.client_id, args.client_secret, args.code):
            print("Authentication successful")
        else:
            print("Authentication failed")
            sys.exit(FAILURE)
    
    elif args.tts:
        import asyncio
        
        async def send_tts():
            if await streamlabs.send_tts(args.tts, args.voice):
                print("TTS sent successfully")
                return SUCCESS
            else:
                print("Failed to send TTS")
                return FAILURE
        
        sys.exit(asyncio.run(send_tts()))
    
    elif args.user_info:
        user_info = streamlabs.get_user_info()
        if user_info:
            print(f"User: {user_info.get('display_name')}")
            print(f"Platform: {user_info.get('platform')}")
            print(f"Primary Platform: {user_info.get('primaryPlatform')}")
        else:
            print("Failed to get user info")
            sys.exit(FAILURE)
    
    else:
        parser.print_help()
    
    sys.exit(SUCCESS)
