# Cleans translations by removing unused keys
# Usage: python clean_translations.py
# Note: Must be run from invokeai/frontend/web/scripts directory
#
# After running the script, open `en.json` and check for empty objects (`{}`) and remove them manually.
# Also, the script does not handle keys with underscores. They need to be checked manually.

import json
import os
import re
from typing import TypeAlias, Union

from tqdm import tqdm

RecursiveDict: TypeAlias = dict[str, Union["RecursiveDict", str]]


class TranslationCleaner:
    file_cache: dict[str, str] = {}

    def _get_keys(self, obj: RecursiveDict, current_path: str = "", keys: list[str] | None = None):
        if keys is None:
            keys = []
        stack = [(obj, current_path)]
        while stack:
            current_obj, current_path = stack.pop()
            for key, value in current_obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, dict):
                    stack.append((value, new_path))
                elif "_" in key:
                    continue
                else:
                    keys.append(new_path)
        return keys

    def _search_codebase(self, key: str):
        key_pattern = re.compile(r"['\"`]" + re.escape(key) + r"['\"`]")
        stem_pattern = re.compile(re.escape(key.split(".")[-1]) + r"['\"`]")

        if not hasattr(self, "_src_files"):
            self._src_files = []
            for root, _dirs, files in os.walk("../src"):
                for file in files:
                    if file.endswith(".ts") or file.endswith(".tsx"):
                        self._src_files.append(os.path.join(root, file))

        for full_path in self._src_files:
            if full_path in self.file_cache:
                content = self.file_cache[full_path]
            else:
                with open(full_path, "r") as f:
                    content = f.read()
                    self.file_cache[full_path] = content

            if key_pattern.search(content):
                return True
            if stem_pattern.search(content):
                return True
        return False

    def _remove_key(self, obj: RecursiveDict, key: str):
        path = key.split(".")
        last_key = path[-1]
        for k in path[:-1]:
            obj = obj[k]
        del obj[last_key]

    def clean(self, obj: RecursiveDict) -> RecursiveDict:
        keys = self._get_keys(obj)
        pbar = tqdm(keys, desc="Checking keys")
        for key in pbar:
            if not self._search_codebase(key):
                self._remove_key(obj, key)
        return obj


def main():
    try:
        with open("../public/locales/en.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Unable to find en.json file - must be run from invokeai/frontend/web/scripts directory"
        ) from e

    cleaner = TranslationCleaner()
    cleaned_data = cleaner.clean(data)

    with open("../public/locales/en.json", "w") as f:
        json.dump(cleaned_data, f, indent=4)


if __name__ == "__main__":
    main()
