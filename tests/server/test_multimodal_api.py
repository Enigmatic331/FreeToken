from __future__ import annotations

from freetoken.server.anthropic_api import convert_anthropic_prompt
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.generation import render_messages
from freetoken.server.responses_api import _convert_input_item


def test_openai_image_parts_preserve_order_for_chat_template():
    messages = render_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    )
    assert messages[0]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": "https://x.test/a.png"},
        {"type": "text", "text": "after"},
    ]


def test_responses_input_image_normalizes_to_chat_image_part():
    messages = _convert_input_item(
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                {"type": "input_text", "text": "describe"},
            ],
        }
    )
    assert messages[0]["content"][0] == {
        "type": "image_url",
        "image_url": "data:image/png;base64,AAAA",
    }


def test_anthropic_base64_image_normalizes_to_data_url():
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA",
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        }
    )
    messages, _, _, _ = convert_anthropic_prompt(req)
    assert messages[0]["content"][0] == {
        "type": "image_url",
        "image_url": "data:image/png;base64,AAAA",
    }
