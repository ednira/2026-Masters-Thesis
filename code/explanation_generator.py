from openai import OpenAI

import base64
import os
from pathlib import Path 
import json


_ROLE = """You are a domain expert in process mining and business process modeling."""

_CONTEXT = """You are given an image of an object-centric process model, with object types represented by colors. Start/end nodes represent object-type lifecycles and are not process activities. In business process modeling, workflow patterns describe behavior between activities."""

_TASK = """Your task is to explain the process model in the image to business users without expertise in process mining. First, for each activity, identify and list its predecessor and successor activities by tracking its edges, use only activity-to-activity relationships. Second, identify the start (no predecessor) and end (no successor) activities in the process. Third, if multiple starts exist, identify where the process most probably begins. Fourth, describe the workflow. Fifth, analyze the workflow to identify if there are bottlenecks and inefficiencies, and to propose process improvement (e.g., combining, eliminating, creating, or reordering activities, among other improvements). Return only valid JSON with this structure: {"explanation":{"summary":"...","workflow description":["...","..."],"inefficiencies":["...","..."],"improvements":["...","..."]}}. Use plain text only, no Markdown. Structure explanation fields "inefficiencies" and "improvements" with bullet points."""

_CONSTRAINTS = """Do not include the steps above or any intermediate reasoning in your output. Return only the JSON object."""

prompt_combination = " ".join([_ROLE, _CONTEXT, _TASK, _CONSTRAINTS])

def image_to_data(path):
    path = Path(path)

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    mime_type = mime_types[path.suffix.lower()]

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return f"data:{mime_type};base64,{b64}"


def call_llm(model, content, API_KEY=str, API_BASE_URL=str):

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )


    return response


def save_json(result, filename, file_path):
    experiment_folder = Path(__file__).parent.parent
    results_folder = experiment_folder / "results"
    results_folder.mkdir(exist_ok=True)

    filepath = results_folder / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_generation(image_path=str, filename=str, API_KEY=str, API_BASE_URL=str):

    prompt = prompt_combination

    model = [
        "google/gemini-3.6-flash"
    ]


    img = image_to_data(image_path)
    image_name = Path(image_path).stem

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": img}}
    ]

    for model in model:
        response = call_llm(
            model=model,
            content=content,
            API_KEY=API_KEY,
            API_BASE_URL=API_BASE_URL
        )

        result = {
            "model": model,
            "image": image_name,
            "prompt":"implementation",
            "answer": json.loads(response.choices[0].message.content),
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }

        filename = (
            f"{image_name}_"
            f"{model.replace('/','_')}_"
            f"explanation.json"
        )

        print('GenAI explanation generated for image:', image_name, 'using model:', model, 'and saved to file:', filename)

        save_json(result, filename, file_path=Path(image_path).parent)



def main():
    print('**************************************')
    print('Gen-AI explanations for process models')
    print('**************************************')


    API_KEY = input("Provide your Gemini API key: ").strip('"')

    API_BASE_URL = input("Provide your API base URL: ").strip('"')

    image_path = input("Enter the path to the process model image (PNG format) : ").strip('"')
    print(f"Image path: {image_path}")

    fileName = os.path.splitext(os.path.basename(image_path))[0]


    run_generation(image_path, fileName, API_KEY, API_BASE_URL)



if __name__ == '__main__':
    main()