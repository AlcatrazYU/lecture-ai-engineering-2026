"""第1回: 素のLLMに質問するスクリプト。

使い方:
    # 1回だけ質問する（質問文を省略すると講座に固有の質問をする）
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask

    # 対話モード（同じ会話の履歴を使って続けて質問する）
    uv run python -m app.ask --chat
    # 対話を終了するには exit または quit を入力する

通常モードは会話履歴を持たない1回限りの呼び出し。
対話モードでは同じチャットセッションを使い、前の質問と回答を踏まえて応答する。

どちらのモードもツールは使わない「素の」呼び出し。
講座に固有のことを聞くと、通常モードでは「分からない」と答えるか、
一般論でもっともらしく答える（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is None:
        print("[usage] トークン使用量を取得できませんでした")
    else:
        print(
            f"[usage] 入力: {usage.prompt_token_count} tokens / "
            f"出力: {usage.candidates_token_count} tokens"
        )


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    _print_usage(response)

    return response.text


def run_chat() -> None:
    """1つのチャットセッションで質問を繰り返し、exit/quit または EOF で終了する。"""
    client = genai.Client(api_key=get_api_key())
    # セッションをループの外で1度だけ作り、各質問で会話履歴を引き継ぐ。
    conversation = client.chats.create(model=MODEL)
    print("対話モードです。終了するには exit または quit を入力してください。")

    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print("\n対話を終了します。")
            return

        if question.lower() in {"exit", "quit"}:
            print("対話を終了します。")
            return
        if not question:
            continue

        response = conversation.send_message(question)
        _print_usage(response)
        print(f"\n{response.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保ったまま対話する（exit または quit で終了）",
    )
    args = parser.parse_args()

    if args.chat and args.question is not None:
        parser.error("--chat と質問文は同時に指定できません")

    print(f"[model] {MODEL}")
    if args.chat:
        try:
            run_chat()
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
