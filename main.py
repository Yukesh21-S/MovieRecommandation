"""Entry point for running the Movie Chatbot backend API server."""

import uvicorn


def main() -> None:
    """Start the FastAPI backend using Uvicorn."""
    uvicorn.run(
        "movie_chatbot.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    # Run this file directly to launch the API server.
    main()