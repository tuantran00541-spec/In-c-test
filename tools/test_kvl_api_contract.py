#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import threading
import unittest

import kvl_api


class FakeRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def prepare(self, messages, system=None, tools=None):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        return kvl_api.PreparedPrompt([101, 102, 103])

    def preflight(self, prepared, max_tokens, temperature):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def generate(self, prepared, max_tokens, temperature, seed, on_delta=None):
        text = "xin chào"
        if on_delta:
            on_delta("xin ")
            on_delta("chào")
        return kvl_api.GenerationResult(text=text, output_tokens=2, stopped_by_eos=True)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = kvl_api.KimiHTTPServer(
            ("127.0.0.1", 0),
            FakeRuntime(),
            kvl_api.DEFAULT_MODEL_ID,
            "local-kimi",
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None, key="local-kimi"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"x-api-key": key}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["content-type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        headers_out = dict(resp.getheaders())
        conn.close()
        return resp.status, headers_out, raw

    def test_health_and_models(self):
        status, _, raw = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["backend"], "serialized-process-v1")

        status, _, raw = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["data"][0]["id"], kvl_api.DEFAULT_MODEL_ID)

    def test_auth(self):
        status, _, _ = self.request("GET", "/healthz", key="wrong")
        self.assertEqual(status, 401)

    def test_anthropic_nonstream_and_count(self):
        req = {
            "model": kvl_api.DEFAULT_MODEL_ID,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, _, raw = self.request("POST", "/v1/messages/count_tokens", req)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["input_tokens"], 3)

        status, _, raw = self.request("POST", "/v1/messages", req)
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["content"][0]["text"], "xin chào")
        self.assertEqual(data["stop_reason"], "end_turn")
        self.assertEqual(data["usage"]["output_tokens"], 2)

    def test_openai_nonstream(self):
        req = {
            "model": kvl_api.DEFAULT_MODEL_ID,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, _, raw = self.request("POST", "/v1/chat/completions", req)
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "xin chào")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")

    def test_anthropic_stream(self):
        req = {
            "model": kvl_api.DEFAULT_MODEL_ID,
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, headers, raw = self.request("POST", "/v1/messages", req)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        text = raw.decode()
        self.assertIn("event: message_start", text)
        self.assertIn('"type":"text_delta","text":"xin "', text)
        self.assertIn('"type":"message_stop"', text)

    def test_openai_stream(self):
        req = {
            "model": kvl_api.DEFAULT_MODEL_ID,
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, headers, raw = self.request("POST", "/v1/chat/completions", req)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        text = raw.decode()
        self.assertIn('"delta":{"content":"xin "}', text)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_dialogue_formatter_preserves_turns_and_warns_tools(self):
        formatted = kvl_api._format_dialogue(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": [{"type": "text", "text": "u2"}]},
            ],
            system="sys",
            tools=[{"name": "Read"}],
        )
        self.assertTrue(formatted.startswith("<|im_system|>system<|im_middle|>sys"))
        self.assertIn("native Anthropic/OpenAI tool calls are not implemented yet", formatted)
        self.assertIn("<|im_user|>user<|im_middle|>u1<|im_end|>", formatted)
        self.assertIn("<|im_assistant|>assistant<|im_middle|>a1<|im_end|>", formatted)
        self.assertTrue(formatted.endswith("<|im_assistant|>assistant<|im_middle|>"))


if __name__ == "__main__":
    unittest.main()
