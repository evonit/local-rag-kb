# SPDX-License-Identifier: Apache-2.0

"""
Description: Prompt management module for loading and retrieving chat prompt templates from a YAML configuration file.
"""



import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from langchain_core.prompts import ChatPromptTemplate

# ── Custom errors ─────────────────────────────────────────────
class PromptFileMissingError(FileNotFoundError):
    pass

class PromptConfigError(ValueError):
    pass

class PromptNotFoundError(KeyError):
    pass

MessageDef = Union[str, List[str]]

class PromptStore:
    """
    엄격 모드: 파일/프롬프트/메시지 블록이 없으면 예외 발생.
    - 파일 없음  -> PromptFileMissingError
    - prompts.<name> 없음 -> PromptNotFoundError
    - 메시지 블록(system/assistant/human) 전부 없음 -> PromptConfigError
    """
    def __init__(self, path: str = "./config/prompts.yaml"):
        self.path = Path(path)
        self._raw: Dict[str, Any] = {}
        self.version: str = "unknown"
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            raise PromptFileMissingError(f"Prompt YAML not found: {self.path}")
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            raise PromptConfigError(f"Failed to parse YAML {self.path}: {e}") from e
        self._raw = data
        self.version = str(data.get("version", "unversioned"))

    def chat_template(self, name: str) -> ChatPromptTemplate:
        prompts = self._raw.get("prompts")
        if not isinstance(prompts, dict):
            raise PromptConfigError("`prompts` section missing or invalid in YAML.")

        block = prompts.get(name)
        if not isinstance(block, dict) or not block:
            raise PromptNotFoundError(f"Prompt '{name}' not found in YAML.")

        msgs = []
        any_added = False
        for role in ("system", "assistant", "human"):
            v: Optional[MessageDef] = block.get(role)
            if v is None:
                continue
            if isinstance(v, list):
                for s in v:
                    msgs.append((role, str(s)))
                    any_added = True
            else:
                msgs.append((role, str(v)))
                any_added = True

        if not any_added:
            raise PromptConfigError(f"Prompt '{name}' has no message blocks (system/assistant/human).")

        return ChatPromptTemplate.from_messages(msgs)

    # (선택) 디버깅용 원본 접근
    def raw_block(self, name: str) -> Dict[str, Any]:
        prompts = self._raw.get("prompts") or {}
        if name not in prompts:
            raise PromptNotFoundError(f"Prompt '{name}' not found in YAML.")
        return prompts[name]


if __name__ == "__main__":
    # 간단한 테스트용 코드
    store = PromptStore("../config/prompts.yaml")
    try:
        template = store.chat_template("clarify")
        print(f"Loaded prompt: {template}")
    except (PromptFileMissingError, PromptConfigError, PromptNotFoundError) as e:
        print(f"Error: {e}")