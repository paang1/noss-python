import json
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import List, Optional
from enum import IntEnum


class EventKind(IntEnum):
    SET_METADATA = 0
    TEXT_NOTE = 1
    RECOMMEND_RELAY = 2
    CONTACTS = 3
    ENCRYPTED_DIRECT_MESSAGE = 4
    DELETE = 5
    REACTION = 7
    BADGE_AWARD = 8
    CHANNEL_CREATE = 40
    CHANNEL_META = 41
    CHANNEL_MESSAGE = 42
    CHANNEL_HIDE = 43
    CHANNEL_MUTE = 44
    REPORT = 1984
    ZAP_REQUEST = 9734
    ZAPPER = 9735
    RELAY_LIST_METADATA = 10002
    PROFILE_BADGES = 30008
    BADGE_DEFINITION = 30009
    LONG_FORM_CONTENT = 30023


@dataclass
class Event:
    """Event class.
    :param content: content string
    :param pukey: public key in hex form
    :param created_at: event creation date
    :param kind: event kind
    :param tags: list of strings
    :param id: event id, will be computed
    :param sig: signature, will be created after signing with a private key
    """

    content: Optional[str] = None
    pubkey: Optional[str] = None
    created_at: Optional[int] = None
    kind: Optional[int] = EventKind.TEXT_NOTE
    tags: List[List[str]] = None
    id: Optional[str] = None
    sig: Optional[str] = None

    def __post_init__(self):
        if self.content is not None and not isinstance(self.content, str):
            # DMs initialize content to None but all other kinds should pass in a str
            raise TypeError("Argument 'content' must be of type str")
        if self.created_at is None:
            self.created_at = int(time.time())

        if self.tags is None:
            self.tags = []

        if self.id is None:
            self.compute_id()

    def serialize(self) -> bytes:
        data = [0, self.pubkey, self.created_at, self.kind, self.tags, self.content]
        data_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return data_str.encode()

    def compute_id(self):
        self.id = sha256(self.serialize()).hexdigest()
