"""質問・対話モードのテスト（実際のGemini APIは呼び出さない）。"""

import sys
from types import SimpleNamespace

from app import ask


def test_run_chat_reuses_one_conversation(monkeypatch, capsys):
    responses = [
        SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=4
            ),
            text="最初の回答",
        ),
        SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5
            ),
            text="前の質問を踏まえた回答",
        ),
    ]

    class FakeConversation:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return responses[len(self.questions) - 1]

    conversation = FakeConversation()

    class FakeChats:
        def create(self, *, model):
            assert model == ask.MODEL
            return conversation

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.chats = FakeChats()

    inputs = iter([" LoRAとは何ですか？ ", "それはなぜですか？", "exit"])
    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask.run_chat()

    output = capsys.readouterr().out
    assert conversation.questions == ["LoRAとは何ですか？", "それはなぜですか？"]
    assert "最初の回答" in output
    assert "前の質問を踏まえた回答" in output
    assert "入力: 10 tokens / 出力: 5 tokens" in output
    assert "対話を終了します。" in output


def test_main_keeps_single_question_mode(monkeypatch, capsys):
    questions = []
    monkeypatch.setattr(sys, "argv", ["app.ask", "単発の質問"])
    monkeypatch.setattr(ask, "ask", lambda question: questions.append(question) or "回答")

    ask.main()

    assert questions == ["単発の質問"]
    assert "回答" in capsys.readouterr().out


def test_main_selects_chat_mode(monkeypatch):
    called = []
    monkeypatch.setattr(sys, "argv", ["app.ask", "--chat"])
    monkeypatch.setattr(ask, "run_chat", lambda: called.append(True))

    ask.main()

    assert called == [True]
