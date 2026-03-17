import os

from ecologits import EcoLogits
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# This is where we could init our lib
EcoLogits.init(providers=['openai'])

MODEL = 'gpt-4.1-nano'

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

total_energy = 0.0
total_emissions = 0.0

messages = []

print(f'Starting a conversation with {MODEL}. Type \'exit\' to quit')

while True:
    prompt = input('\nInput your prompt: ')

    if prompt.lower() == 'exit':
        print('\nExiting...')
        break

    messages.append({'role': 'user', 'content': prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content

    print(f'\n{MODEL}: {reply}')

    messages.append({'role': 'assistant', 'content': reply})

    energy, emissions = response.impacts.energy.value, response.impacts.gwp.value

    total_energy += energy
    total_emissions += emissions

    print(f"\nEnergy this request: {energy} kWh")
    print(f"Total energy used: {total_energy} kWh")

    print(f"GHG this request: {emissions} kgCO2eq")
    print(f"Total GHG emissions: {total_emissions} kgCO2eq")

    # Get potential warnings
    if response.impacts.has_warnings:
        for w in response.impacts.warnings:
            print(w)

    # Get potential errors
    if response.impacts.has_errors:
        for e in response.impacts.errors:
            print(e)
