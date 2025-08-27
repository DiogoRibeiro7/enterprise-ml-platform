"""Notification client implementations."""

from .email import EmailClient
from .slack import SlackClient

__all__ = ["EmailClient", "SlackClient"]
