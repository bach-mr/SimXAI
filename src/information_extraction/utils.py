import json


def load_json(path):
    jsonFile = open(path, "r", encoding='utf-8')
    jsonContent = jsonFile.read()
    content_ls = json.loads(jsonContent)
    jsonFile.close()

    return content_ls


def save_json(output, path):
    jsonString = json.dumps(output, ensure_ascii=False)
    jsonFile = open(path, "w", encoding='utf-8')
    jsonFile.write(jsonString)
    jsonFile.close()