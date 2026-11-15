import json
def tools_prompt(functions, question):
    return ("You have access to the following functions (JSON specs):\n" + json.dumps(functions, ensure_ascii=False) +
            "\n\nUser request: " + question.strip() +
            "\n\nAnswer with only the function call(s) as a JSON list inside a ```json block, in this format: "
            '[{"name": "<function name>", "arguments": {"<param>": <value>}}]. '
            "Use one list item per call (several items if several calls are needed). Use exact function and parameter names from the specs.")
def tools_answer(calls):
    return "```json\n" + json.dumps([{"name": c["name"], "arguments": c["arguments"]} for c in calls], ensure_ascii=False) + "\n```"
