from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
    model='deepseek-v2',
    messages=[
        {
            'role': 'user',
            'content': '''You are playing a grid-based board game on a 6x6 square grid. Various objects are placed on the board, 
            and your goal is to complete a specific mission by reaching a target object.

            I will provide you with:
            - The mission objective
            - A description of your surroundings
            - The available moves you can make

            Valid actions are: left, right, forward, pickup, drop, toggle, done

            After each move you choose, I will update you with the new state of the environment. Continue selecting moves until you complete the mission.

            To complete the mission, you must move to a position directly next to the target object and face it.

            Mission: put the blue box next to the yellow key
            BabyAI-Text sees:
            - You see a wall 2 steps left
            - You see a yellow key 2 steps forward
            - You see a purple key 2 steps right and 1 step forward
            - You see a purple key 2 steps right

            Only respond with one word: either "left", "right", "forward", "pickup", "drop", "toggle", or "done".
            Do not explain your reasoning. Just return the selected move.
            ''',
        },
    ]
)

print(response['message']['content'])



