import requests

BACKEND_URL = "http://localhost:8000/ask"

def main():
    print("\nLocal AI Search CLI")
    print("Type 'exit' to quit\n")

    while True:
        query = input("AI Search > ")

        if query.lower() in ["exit", "quit"]:
            break

        try:
            response = requests.get(
                BACKEND_URL,
                params={"q": query}
            )

            data = response.json()

            print("\nANSWER:\n")
            print(data["data"])

            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()