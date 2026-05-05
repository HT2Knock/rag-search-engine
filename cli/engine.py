import json
import keyword
import string


class SearchEngine:
    def search(self, path: str, query: str):
        try:
            with open(path, "r") as file:
                data = json.load(file)
                count = 0

                for movie in data.get("movies"):
                    id = movie.get("id")
                    title = movie.get("title")

                    isMatch = False
                    for word in query.split():
                        for keyword in self.process_string(title):
                            if word in keyword:
                                isMatch = True

                    if not isMatch:
                        continue

                    print(f"{id}. {title}")
                    count += 1
                    if count > 5:
                        break
        except:
            raise RuntimeError("failed to get movies")

    def process_string(self, input: str) -> list[str]:
        return self.remove_punctuation(input.lower()).split()

    def remove_punctuation(self, input: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return input.translate(table)
