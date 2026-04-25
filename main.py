"""
main.py
Entry point for the Movie Chatbot.

Run:  python main.py
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _ensure_db_ready() -> None:
    """Build the vector DB on first run; skip if already populated."""
    import vector_engine as vdb
    import data_fetcher as tmdb

    col = vdb.get_collection()
    if col and col.count() > 0:
        print(f"[Setup] Vector DB ready ({col.count()} movies).")
        return

    print("[Setup] First run — building vector database…")
    if not os.path.exists("movies_data.json"):
        tmdb.fetch_and_save_diverse_movies()
    vdb.setup_vector_db()
    print("[Setup] Done.")


def main() -> None:
    _ensure_db_ready()

    from agent import run  # import after DB is ready

    history: list[dict] = []

    print("\n" + "=" * 50)
    print("   🎬  Movie Chatbot (LangGraph + Groq)  🎬")
    print("=" * 50)
    print("  Ask anything, e.g. 'latest Tamil movies'")
    print("  Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 🎬")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 🎬")
            break

        response, history = run(query, history)
        print(f"\nChatbot: {response}\n")


if __name__ == "__main__":
    main()