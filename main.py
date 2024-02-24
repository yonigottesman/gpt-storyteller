import json
import random

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

app = FastAPI()


templates = Jinja2Templates(directory="templates")

# this list was generate by chatgpt
topics = [
    "Adventure",
    "Pirates",
    "Space exploration",
    "Magic",
    "Dragons",
    "Superheroes",
    "Time travel",
    "Mystery",
    "Dinosaurs",
    "Robots",
    "Underwater worlds",
    "Wizards",
    "Friendship",
    "Treasure hunts",
    "Ghost stories",
    "Animal heroes",
    "Fairy tales",
    "Sports",
    "Science fiction",
    "Fantasy realms",
    "Secret societies",
    "Mythical creatures",
    "Survival stories",
    "Historical adventures",
    "Alien encounters",
    "School life",
    "Jungle adventures",
    "Arctic expeditions",
    "Medieval quests",
    "Vampire legends",
    "Witchcraft",
    "Parallel universes",
    "Ninja warriors",
    "Spy missions",
    "Circus life",
    "Zombie apocalypses",
    "Lost civilizations",
    "Eco adventures",
    "Racing competitions",
    "Haunted houses",
    "Super villains",
    "Secret agents",
    "Monster hunters",
    "Enchanted forests",
    "Virtual reality",
    "Inventors and inventions",
    "Ancient myths",
    "Desert islands",
    "Space colonies",
    "Magical kingdoms",
    "Underdog stories",
    "Detective cases",
    "Wild West",
    "Futuristic cities",
    "Boarding schools",
    "Animal kingdoms",
    "Time capsules",
    "Lost treasures",
    "Cyber adventures",
    "Dream worlds",
    "Parallel dimensions",
    "Super powers",
    "Ancient Egypt",
    "Knights and castles",
    "Deep sea mysteries",
    "Daring rescues",
    "Hidden worlds",
    "Extreme sports",
    "Forbidden lands",
    "Alien planets",
    "Ancient Rome",
    "Warrior princesses",
    "Jungle tribes",
    "Arctic mysteries",
    "Mythical islands",
    "Ghost towns",
    "Musical adventures",
    "Video game worlds",
    "Space stations",
    "Dragon riders",
    "Lost temples",
    "Magical artifacts",
    "Soccer tournaments",
    "Camping trips",
    "Piano recitals",
    "Dragon riders",
]

SYSTEM_PROMPT = "You make up cool and educational stories for children aged 7-16 in hebrew. "
PROMPT = """make up a story in hebrew about 20-50 lines long. return a json with 'title' 'text' and 'questions'.
the questions is a list of 5 questions on the story. all in hebrew.
The topic should be any combination of one or more of the following: [{topic_1},{topic_2},{topic_3}]."""
MODEL = "gpt-4-turbo-preview"


@app.get("/gen")
async def gen():

    chosen_topics = random.sample(topics, 3)
    print(chosen_topics)
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        response_format={"type": "json_object"},
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT.format(topic_1=chosen_topics[0], topic_2=chosen_topics[1], topic_3=chosen_topics[2]),
            },
        ],
    )
    story = json.loads(response.choices[0].message.content)
    print(story["title"])
    dalle_response = await client.images.generate(
        model="dall-e-3",
        prompt=f"generate an image for this short story:\n{story['text']}",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return {
        "text": story["text"],
        "title": story["title"],
        "questions": story["questions"],
        "image_url": dalle_response.data[0].url,
    }


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
