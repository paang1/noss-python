import json
import time
from dataclasses import dataclass
import random
import subprocess
from typing import List, Optional
from enum import IntEnum
from loguru import logger


def hash_event(envet_str: str):
    executable = r".\hash.exe"
    arg2 = str(random.randint(1000, 9999))
    result = subprocess.run([executable, envet_str, arg2], capture_output=True, text=True)
    hash_hex = result.stdout.split('\n')[0].strip()
    return hash_hex


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

    def format_event(self):
        # 使用 self 直接访问属性
        content = json.dumps(self.content if self.content is not None else {}, separators=(',', ':'))
        pubkey = self.pubkey if self.pubkey is not None else ''
        created_at = self.created_at if self.created_at is not None else 0
        kind = self.kind if self.kind is not None else 0
        tags = self.tags if self.tags is not None else []
        nonce = self.nonce if hasattr(self, 'nonce') else '{nonce}'

        # 处理 tags，只保留值
        formatted_tags = [[str(item) for item in sublist] for sublist in tags]

        # 组合所有部分
        formatted_event = f'[{kind},{created_at},{formatted_tags},{content},"{pubkey}"]'

        return formatted_event

    # def serialize(self) -> bytes:
    #     data = [0, self.pubkey, self.created_at, self.kind, self.tags, self.content]
    #     data_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    #     return data_str.encode()

    # def compute_id(self):
    #     self.id = sha256(self.serialize()).hexdigest()
    def compute_id(self):
        formatted_event_string = self.format_event()
        self.id = hash_event(envet_str=formatted_event_string)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pubkey": self.pubkey,
            "created_at": self.created_at,
            "kind": self.kind,
            "tags": self.tags,
            "content": self.content,
            "sig": self.sig,
        }

