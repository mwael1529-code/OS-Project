"""Interactive text-only demo for the Voice Ubuntu NLP v6 parser."""
import json
from voice_ubuntu_parser import parse_text


def main():
    print("Voice Ubuntu NLP v6 text demo. Type 'exit' to stop.")
    print("Examples: Open Firefox | Search YouTube for Python | Play Interstellar")
    while True:
        try:
            text = input("Command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if text.lower() == "exit":
            break
        result = parse_text(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
