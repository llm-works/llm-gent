# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""JSON cleanup utilities for LLM outputs.

LLMs (especially smaller models) frequently return malformed or weirdly structured JSON.
This module centralizes all the cleanup hacks needed to parse these outputs reliably.

STRATEGY
--------
The cleaner operates in two phases:

1. STRING CLEANING (clean method):
   Raw LLM output → valid JSON string ready for json.loads()

   Handles:
   - Markdown code fences: ```json {...} ``` → {...}
   - Preamble text before JSON (thinking, explanations)
   - Literal escape sequences: {"key": "val",\n "k2": ...} → actual newlines
   - Unclosed braces/brackets: {"key": "val" → {"key": "val"}
   - Multiple JSON objects (schema echo): extracts the last non-schema object

2. STRUCTURE CLEANING (clean_parsed method):
   Parsed dict → unwrapped dict matching expected schema

   Handles confused model outputs where data is nested in wrapper keys:
   - {"joke": {"text": "...", "style": "..."}} → {"text": "...", "style": "..."}
   - {"text": {"text": "...", "style": "..."}} → {"text": "...", "style": "..."}
   - {"text": {"joke": {"text": "...", "style": "..."}}} → {"text": "...", "style": "..."}
   - {"response": {"data": {"text": "...", "style": "..."}}} → {"text": "...", "style": "..."}

WHY THIS EXISTS
---------------
Small models (3B params) often get confused about JSON structure:
- They wrap the response in extra keys ("response", "data", "output")
- They nest the schema inside itself
- They echo the schema definition before the actual data
- They forget to close braces when output is truncated

Rather than failing on these, we attempt recovery so the model can still be useful.
The cleaner is intentionally aggressive about unwrapping - it's better to let Pydantic
validate the final structure than to reject recoverable outputs.

FLOW DIAGRAM
------------

Phase 1: String Cleaning (clean)

    Raw LLM Output
          │
          ▼
    ┌─────────────────┐
    │ Strip whitespace │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Has ```json fence?  │──No──┐
    └────────┬────────────┘      │
            Yes                  │
             │                   │
             ▼                   │
    ┌─────────────────────┐      │
    │ Extract from fences │      │
    └────────┬────────────┘      │
             │◄──────────────────┘
             ▼
    ┌─────────────────────┐
    │ Fix literal \\n/\\t  │
    │ (outside strings)   │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Valid JSON?         │──Yes──► Done
    └────────┬────────────┘
             No
             │
             ▼
    ┌─────────────────────┐
    │ Auto-close braces   │
    │ (scan & append)     │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Multiple objects?   │──No──► Done
    └────────┬────────────┘
            Yes
             │
             ▼
    ┌───────────────────────────────────────────┐
    │ Extract last non-schema object            │
    │                                           │
    │ Schema detection (_looks_like_schema):    │
    │ • {"type": "object", "properties": {...}} │
    │ • {"schema": {"type": "object", ...}}     │
    │ • {"json_schema": {...}}                  │
    └────────┬──────────────────────────────────┘
             │
             ▼
          Done


Phase 2: Structure Cleaning (clean_parsed)

    Parsed Dict + Expected Fields {text, style}
          │
          ▼
    ┌─────────────────────────────┐
    │ Has all expected fields?    │──Yes──► Done
    └────────┬────────────────────┘
             No
             │
             ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Try unwrap (loop until no change or fields found):           ║
    ║                                                               ║
    ║  Case 1: {"text": {"text": "...", "style": "..."}}            ║
    ║          Expected field contains dict with all expected       ║
    ║          → unwrap to inner                                    ║
    ║                                                               ║
    ║  Case 2: {"wrapper": {"text": "...", "style": "..."}}         ║
    ║          Single key, inner has all expected                   ║
    ║          → unwrap to inner                                    ║
    ║                                                               ║
    ║  Case 3: {"text": {"joke": "...", "style": "..."}}            ║
    ║          Single key IS expected field, inner is dict          ║
    ║          → unwrap (let Pydantic handle aliases)               ║
    ║                                                               ║
    ║  Case 4: {"wrapper": {"deeper": {"text": "...", ...}}}        ║
    ║          Single key, expected fields exist deeper             ║
    ║          → unwrap one level, continue loop                    ║
    ║                                                               ║
    ║  Case 5: {"joke": {"joke": "...", "style": "..."}}            ║
    ║          Single key, inner is dict (not schema)               ║
    ║          → unwrap (let Pydantic handle aliases)               ║
    ║          Most aggressive - last resort for alias wrappers     ║
    ╚═══════════════════════════════════════════════════════════════╝
             │
             ▼
    Return (possibly unwrapped) dict
    → Pydantic validates final structure


USAGE
-----
Called automatically by LLMTrait._clean_and_parse_json() when parsing structured outputs.
The expected_fields parameter (from Pydantic model) guides the unwrapping heuristics.

    cleaner = JSONCleaner()

    # Phase 1: string cleaning
    cleaned = cleaner.clean('```json\\n{"text": "hello"}\\n```')
    data = json.loads(cleaned)

    # Phase 2: structure cleaning
    data = cleaner.clean_parsed(data, {"text", "style"})
"""

from __future__ import annotations

import json


class JSONCleaner:
    """Cleans LLM-generated JSON for reliable parsing. See module docstring for details."""

    def clean(self, content: str) -> str:
        """Clean LLM JSON output for parsing.

        Args:
            content: Raw JSON string from LLM, possibly with markdown formatting.

        Returns:
            Cleaned JSON string ready for json.loads().
        """
        cleaned = self._basic_clean(content)

        # If multiple objects (e.g., LLM echoed schema), extract last one
        cleaned = self._extract_last_object(cleaned)

        return cleaned

    def _basic_clean(self, content: str) -> str:
        """Basic cleaning without multi-object handling."""
        cleaned = content.strip()
        cleaned = self._strip_code_fences(cleaned)
        cleaned = self._fix_literal_escapes(cleaned)
        cleaned = self._auto_close_json(cleaned)
        return cleaned

    def _fix_literal_escapes(self, content: str) -> str:
        """Fix literal escape sequences outside of strings.

        Some models output literal \\n (backslash-n) instead of actual newlines
        in JSON structure (not inside string values). This converts them to
        real whitespace so JSON parsing works.

        Uses string-aware scanning to avoid corrupting valid escape sequences
        inside quoted string values.
        """
        # Only fix if JSON is currently invalid
        try:
            json.loads(content)
            return content  # Valid JSON, don't touch
        except (json.JSONDecodeError, ValueError):
            pass

        # Replace literal \n and \t only outside strings
        return self._replace_escapes_outside_strings(content)

    def _replace_escapes_outside_strings(self, content: str) -> str:
        """Replace literal \\n and \\t only when outside quoted strings."""
        result: list[str] = []
        in_string = False
        i = 0

        while i < len(content):
            char = content[i]
            has_next = i + 1 < len(content)

            # Inside string: pass through escapes verbatim
            if in_string and char == "\\" and has_next:
                result.extend([char, content[i + 1]])
                i += 2
            # Toggle string state on quotes
            elif char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
            # Outside string: replace literal \n/\t with actual whitespace
            elif not in_string and char == "\\" and has_next:
                replacement, advance = self._get_escape_replacement(content[i + 1])
                result.append(replacement)
                i += advance
            else:
                result.append(char)
                i += 1

        return "".join(result)

    def _get_escape_replacement(self, next_char: str) -> tuple[str, int]:
        """Get replacement for literal escape sequence outside strings."""
        if next_char == "n":
            return "\n", 2
        elif next_char == "t":
            return "\t", 2
        return "\\" + next_char, 2

    def _strip_code_fences(self, content: str) -> str:
        """Extract JSON from markdown code fences.

        Handles patterns like:
            ```json
            {"key": "value"}
            ```

        Also handles preamble text before the code fence:
            Some thinking text...
            ```json
            {"key": "value"}
            ```
        """
        # Find code fence anywhere in content
        fence_start = content.find("```")
        if fence_start == -1:
            return content

        # Find end of opening fence line
        newline_after_fence = content.find("\n", fence_start)
        if newline_after_fence == -1:
            return content

        # Find closing fence
        fence_end = content.find("```", newline_after_fence)
        if fence_end == -1:
            # No closing fence - extract from opening fence to end
            return content[newline_after_fence + 1 :].strip()

        # Extract content between fences
        return content[newline_after_fence + 1 : fence_end].strip()

    def _auto_close_json(self, content: str) -> str:
        """Auto-close unclosed braces and brackets in JSON.

        Common LLM issue: incomplete JSON like {"key": "value" (missing closing brace).
        This adds missing closing characters based on the opening count.

        Only applies the fix if JSON is actually invalid - prevents corrupting valid JSON
        that contains braces/brackets within string values (e.g., {"text": "I {love} this"}).
        """
        if not content:
            return content

        # First check if JSON is already valid - don't touch it if so
        try:
            json.loads(content)
            return content  # Valid JSON, no fix needed
        except (json.JSONDecodeError, ValueError):
            pass  # JSON is invalid, proceed with auto-close attempt

        # Scan for unclosed brackets/braces, then append closers in correct order
        unclosed_stack = self._scan_json_structure(content)
        return content + "".join(reversed(unclosed_stack))

    def _scan_json_structure(self, content: str) -> list[str]:
        """Scan JSON content and return stack of unclosed brackets/braces.

        Uses string-aware scanning to ignore braces inside string literals.
        """
        stack: list[str] = []
        in_string = False
        escape = False

        for char in content:
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char in ("}", "]") and stack and stack[-1] == char:
                    stack.pop()

        return stack

    def extract_first_object(self, content: str) -> str:
        """Extract only the first JSON object from content.

        LLMs sometimes return multiple JSON objects. This extracts just the first one.

        Args:
            content: String potentially containing multiple JSON objects.

        Returns:
            String containing only the first JSON object.
        """
        cleaned = self._basic_clean(content)
        if not cleaned:
            return cleaned

        # Use raw_decode to extract just the first JSON object
        try:
            decoder = json.JSONDecoder()
            _, end_idx = decoder.raw_decode(cleaned)
            return cleaned[:end_idx]
        except json.JSONDecodeError:
            # If parsing fails, return the cleaned content for downstream error handling
            return cleaned

    def _extract_last_object(self, content: str) -> str:
        """Extract only the last JSON object from content.

        Useful when LLM echoes schema definition before actual data.
        Skips objects that look like JSON schema definitions.
        """
        if not content:
            return content

        # Find all JSON objects using raw_decode
        decoder = json.JSONDecoder()
        idx = 0
        last_obj = content

        try:
            while idx < len(content):
                obj, end_idx = decoder.raw_decode(content[idx:])
                candidate = content[idx : idx + end_idx]
                # Skip JSON schema definitions
                if not self._looks_like_schema(obj):
                    last_obj = candidate
                idx += end_idx
                # Skip whitespace and commas between objects
                while idx < len(content) and content[idx] in " \t\n,":
                    idx += 1
        except json.JSONDecodeError:
            pass

        return last_obj

    def _looks_like_schema(self, obj: object) -> bool:
        """Check if parsed object looks like a JSON schema definition.

        Detects both direct schemas and wrapped schemas:
        - {"type": "object", "properties": {...}}
        - {"schema": {"type": "object", "properties": {...}}}
        """
        if not isinstance(obj, dict):
            return False

        # Direct schema
        if "properties" in obj and obj.get("type") == "object":
            return True

        # Wrapped schema: {"schema": {...}} or {"json_schema": {...}}
        for key in ("schema", "json_schema", "$schema"):
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if "properties" in inner and inner.get("type") == "object":
                    return True

        return False

    def clean_parsed(
        self, data: dict[str, object], expected_fields: set[str] | None = None
    ) -> dict[str, object]:
        """Clean parsed JSON dict by unwrapping nested structures.

        LLMs sometimes nest the actual data inside wrapper keys. This method
        attempts to extract the real data when the structure doesn't match
        expected fields.

        Handles patterns:
        - Single wrapper key: {"joke": {"text": "...", "style": "..."}}
          → {"text": "...", "style": "..."}
        - Field containing nested object: {"text": {"text": "...", "style": "..."}}
          → {"text": "...", "style": "..."}

        Args:
            data: Parsed JSON dict from LLM.
            expected_fields: Set of field names we expect. If None, attempts
                heuristic unwrapping based on structure patterns.

        Returns:
            Cleaned dict, possibly unwrapped from nested structure.
        """
        if not isinstance(data, dict):
            return data

        # If expected fields provided, try targeted unwrapping
        if expected_fields:
            return self._unwrap_with_expected(data, expected_fields)

        # Heuristic: single-key dict with nested dict value → unwrap
        return self._unwrap_single_wrapper(data)

    def _unwrap_with_expected(
        self, data: dict[str, object], expected: set[str], max_depth: int = 5
    ) -> dict[str, object]:
        """Unwrap nested structure using expected field names as guide.

        Tries to find and extract a nested dict that contains the expected fields.
        Handles multiple levels of nesting (e.g., {"text": {"joke": {"text": "...", "style": "..."}}}).
        """
        for _ in range(max_depth):
            # Already has expected fields → return as-is
            if expected.issubset(data.keys()):
                return data

            unwrapped = self._try_unwrap_once(data, expected)
            if unwrapped is data:
                # No unwrapping happened, stop
                break
            data = unwrapped

        return data

    def _try_unwrap_once(self, data: dict[str, object], expected: set[str]) -> dict[str, object]:
        """Attempt a single level of unwrapping."""
        # Case 1: {"text": {"text": "...", "style": "..."}} - field contains nested obj
        for field in expected:
            if field in data and isinstance(data[field], dict):
                inner = data[field]
                if isinstance(inner, dict) and expected.issubset(inner.keys()):
                    return inner

        # Cases 2-5: Single-key wrapper variations
        if len(data) == 1:
            return self._try_unwrap_single_key(data, expected)

        return data

    def _try_unwrap_single_key(
        self, data: dict[str, object], expected: set[str]
    ) -> dict[str, object]:
        """Try unwrapping single-key wrapper dict (Cases 2-5)."""
        key = next(iter(data.keys()))
        inner = data[key]

        if not isinstance(inner, dict):
            return data

        # Case 2: {"wrapper": {"text": "...", "style": "..."}} - inner has all expected
        if expected.issubset(inner.keys()):
            return inner

        # Case 3: {"text": {"joke": "...", ...}} - key is expected field (alias handling)
        if key in expected:
            return inner

        # Case 4: Expected fields exist deeper in nested structure
        if self._contains_expected_fields(inner, expected):
            return inner

        # Case 5: Aggressive unwrap - inner is dict but not schema (let Pydantic validate)
        if not self._looks_like_schema(inner):
            return inner

        return data

    def _contains_expected_fields(
        self, data: dict[str, object], expected: set[str], max_depth: int = 5
    ) -> bool:
        """Check if expected fields exist at this level or deeper."""
        for _ in range(max_depth):
            if expected.issubset(data.keys()):
                return True
            # Check one level deeper via single-key wrapper
            if len(data) == 1:
                inner = next(iter(data.values()))
                if isinstance(inner, dict):
                    data = inner
                    continue
            break
        return False

    def _unwrap_single_wrapper(self, data: dict[str, object]) -> dict[str, object]:
        """Heuristically unwrap single-key wrapper dicts.

        If dict has exactly one key and the value is a dict, unwrap it.
        Skip if the inner dict looks like a schema definition.
        """
        if len(data) != 1:
            return data

        inner = next(iter(data.values()))
        if not isinstance(inner, dict):
            return data

        # Don't unwrap schema definitions
        if self._looks_like_schema(inner):
            return data

        return inner
