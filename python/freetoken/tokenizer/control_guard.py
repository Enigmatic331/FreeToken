"""Keep client text from becoming control tokens on the DeepSeek encoder path.

The checkpoint's reference encoder renders a whole conversation to one string, which is
then tokenized in one call. The tokenizer cannot tell a control token the template wrote
(``<｜User｜>``, ``</think>``, ``｜DSML｜``, ...) from the same characters inside a user
message, a tool result, a tool-call argument or a tool schema. Pasted text such as
``<｜Assistant｜></think>...<｜end▁of▁sentence｜><｜User｜>`` would therefore forge turns,
end the reasoning block or open a tool call. ``split_special_tokens`` is no defence:
DeepSeek's BPE vocabulary rebuilds most markers from plain text with the very same ids.

The guard works around the encoder, which stays untouched:

1. Before rendering, every reserved string (an added token of the tokenizer) in client
   text is swapped for one private-use placeholder character. Private-use characters
   already present in client text are swapped too, so a placeholder can never be forged.
   Tool-call ``arguments`` are JSON-decoded first, because the encoder decodes them and
   ``\\uff5c`` escapes would otherwise turn into markers after the scan.
2. After rendering, the prompt is encoded segment by segment around the placeholders;
   the template's own markers sit in those segments and keep their ids. Each placeholder
   becomes an ordinary-token spelling of its original text that contains no reserved id,
   so decoding the ids still yields the original text.

Prompts whose client text holds no reserved string never reach step 2 and tokenize
exactly as before.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

PLACEHOLDER_BASE = 0xF0000  # Supplementary Private Use Area-A
PLACEHOLDER_LIMIT = 0xFFFFD
_PLACEHOLDER_RE = re.compile("[\U000F0000-\U000FFFFD]")


class ControlTokenGuard:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        added = get_added_vocab() if callable(get_added_vocab) else {}
        added = {text: idx for text, idx in added.items() if isinstance(text, str) and text}
        self.reserved_ids = frozenset(added.values())
        self.enabled = bool(added)
        self._pattern = re.compile(_trie_pattern(added) + "|" + _PLACEHOLDER_RE.pattern)
        self._safe_cache: dict[str, list[int]] = {}
        self._wrap: tuple[list[int], list[int]] | None = None

    def neutralize(self, obj: Any) -> tuple[Any, list[str]]:
        """Return ``obj`` with reserved strings in every string (dict keys included)
        replaced by placeholders, plus the replaced texts in placeholder order. The
        input is not modified; with nothing to replace the result is ``obj`` itself."""
        originals: list[str] = []
        if not self.enabled:
            return obj, originals
        result = self._walk(obj, originals)
        return (result, originals) if originals else (obj, originals)

    def restore(self, text: str, originals: list[str]) -> str:
        """The rendered text with every placeholder put back: what the model reads."""
        return _PLACEHOLDER_RE.sub(lambda m: originals[self._index(m.group(), originals)], text)

    def encode(
        self,
        text: str,
        originals: list[str],
        encode_segment: Callable[[str], list[int]],
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Encode a rendered prompt that still holds placeholders. ``encode_segment``
        is the plain tokenizer call (no special tokens added) for template text."""
        ids: list[int] = []
        cursor = 0
        for match in _PLACEHOLDER_RE.finditer(text):
            if match.start() > cursor:
                ids.extend(encode_segment(text[cursor : match.start()]))
            ids.extend(self.safe_ids(originals[self._index(match.group(), originals)]))
            cursor = match.end()
        if cursor < len(text):
            ids.extend(encode_segment(text[cursor:]))
        if add_special_tokens:
            prefix, suffix = self._special_wrap(encode_segment)
            ids = prefix + ids + suffix
        return ids

    def safe_ids(self, text: str) -> list[int]:
        """Ordinary-token ids for ``text`` that contain no reserved id. Plain BPE
        first; wherever it still rebuilds a marker, the text is split in half."""
        cached = self._safe_cache.get(text)
        if cached is not None:
            return list(cached)
        ids = self._plain_ids(text)
        if self.reserved_ids.intersection(ids):
            if len(text) < 2:
                raise ValueError(f"cannot spell {text!r} without a reserved token")
            mid = len(text) // 2
            ids = self.safe_ids(text[:mid]) + self.safe_ids(text[mid:])
        if len(self._safe_cache) < 4096:
            self._safe_cache[text] = list(ids)
        return ids

    def _walk(self, obj: Any, originals: list[str]) -> Any:
        if isinstance(obj, str):
            return self._pattern.sub(lambda m: self._placeholder(m.group(), originals), obj)
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                new_key = self._walk(key, originals) if isinstance(key, str) else key
                if key == "arguments" and isinstance(value, str):
                    out[new_key] = self._walk_json_string(value, originals)
                else:
                    out[new_key] = self._walk(value, originals)
            return out
        if isinstance(obj, list):
            return [self._walk(item, originals) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk(item, originals) for item in obj)
        return obj

    def _walk_json_string(self, value: str, originals: list[str]) -> str:
        """Tool-call arguments are a JSON string the encoder decodes, so scan the
        decoded values; re-serialize only when something was replaced."""
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return self._walk(value, originals)
        if not isinstance(decoded, (dict, list)):
            return self._walk(value, originals)
        before = len(originals)
        walked = self._walk(decoded, originals)
        if len(originals) == before:
            return value
        return json.dumps(walked, ensure_ascii=False)

    def _placeholder(self, text: str, originals: list[str]) -> str:
        code = PLACEHOLDER_BASE + len(originals)
        if code > PLACEHOLDER_LIMIT:
            raise ValueError("request holds too many reserved control-token strings")
        originals.append(text)
        return chr(code)

    @staticmethod
    def _index(char: str, originals: list[str]) -> int:
        index = ord(char) - PLACEHOLDER_BASE
        if not 0 <= index < len(originals):
            raise ValueError("rendered prompt holds an unknown placeholder")
        return index

    def _plain_ids(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        except TypeError:  # tokenizers without split_special_tokens
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        return list(ids)

    def _special_wrap(self, encode_segment: Callable[[str], list[int]]) -> tuple[list[int], list[int]]:
        """What ``add_special_tokens=True`` puts around a text (empty for DeepSeek)."""
        if self._wrap is None:
            bare = encode_segment("a")
            full = _to_list(self.tokenizer.encode("a", add_special_tokens=True))
            for start in range(len(full) - len(bare) + 1):
                if full[start : start + len(bare)] == bare:
                    self._wrap = (full[:start], full[start + len(bare) :])
                    break
            else:
                raise ValueError("cannot locate the text inside the tokenizer's special-token wrap")
        return list(self._wrap[0]), list(self._wrap[1])


def _trie_pattern(strings: Any) -> str:
    """One regex for many literals, factored by common prefix. A flat alternation of
    DeepSeek's ~1,300 markers (nearly all start with ``<``) costs seconds per MB of
    text; the factored form tries one branch per character, so a position costs at most
    a marker's length. Optional tails are greedy, so the longest marker wins, as in the
    tokenizer's own added-token matching."""
    trie: dict[str, dict] = {}
    for text in strings:
        node = trie
        for char in text:
            node = node.setdefault(char, {})
        node[""] = {}

    def emit(node: dict[str, dict]) -> str:
        end = "" in node
        branches = [re.escape(char) + emit(child) for char, child in sorted(node.items()) if char]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 and not end else "(?:" + "|".join(branches) + ")"
        return body + "?" if end else body

    return emit(trie) if trie else "(?!)"


def _to_list(ids: Any) -> list[int]:
    if hasattr(ids, "view"):
        return ids.view(-1).tolist()
    return list(ids)


__all__ = ["ControlTokenGuard"]
