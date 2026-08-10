import json

def epsilon_key(eps):
    return f"{eps:g}"
def value_tag(value):
    return f"{value:g}".replace(".", "p")
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=float) + "\n"
    )
    temporary.replace(path)


    