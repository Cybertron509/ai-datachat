"""Authentication utilities"""
import hashlib
import json
from pathlib import Path

class AuthManager:
    """Handle user authentication"""
    
    def __init__(self, users_file="users.json"):
        self.users_file = Path(users_file)
        if not self.users_file.exists():
            self._create_default_users()
    
    def _create_default_users(self):
        """Create default users file"""
        default_users = {
            "admin": {
                "password": self._hash_password("YopakaBarem26$"),
                "name": "Administrator",
                "role": "admin"
            }
        }
        self._save_users(default_users)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self) -> dict:
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_users(self, users: dict):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user"""
        users = self._load_users()
        if username in users:
            hashed_password = self._hash_password(password)
            return users[username]["password"] == hashed_password
        return False
    
    def register_user(self, username: str, password: str, name: str, role: str = "user") -> bool:
        """Register new user"""
        users = self._load_users()
        if username in users:
            return False
        
        users[username] = {
            "password": self._hash_password(password),
            "name": name,
            "role": role
        }
        self._save_users(users)
        return True
    
    def get_user_info(self, username: str) -> dict:
        """Get user information"""
        users = self._load_users()
        return users.get(username, {})