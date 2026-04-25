"""Run the Movie Chatbot backend API server."""

import uvicorn


def main() -> None:
    uvicorn.run("movie_chatbot.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()