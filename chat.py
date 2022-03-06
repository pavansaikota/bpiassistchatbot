import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from stack import Stack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
flowStack = Stack()
flowState = {}

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BPI"

def get_response(msg):
    if msg == "cancel":
        return cancelFlow()

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                resp = random.choice(intent['responses'])
                flow = isFlow(resp)
                if len(flow)>0:
                    beginFlow(flow)
                    print(resp.split(':')[2])
                else:
                    return resp
    
    if flowStack.getSize() > 0:
        currFlow = flowStack.peek()
        return processFlow(currFlow,msg)    
    return "I do not understand..."

def isFlow(resp):
    if resp.startswith('Flow'):
        x = resp.split(":")
        flow = x[1]
        return flow
    return ""

def beginFlow(flow):
    flowStack.push(flow)
    flowState[flow] = {'curr':-1,'max':5,'fields':['type','name','building','rack','order'],'endMsg':'Device is created','result':{}}

def processFlow(flow,msg):
    state = flowState[flow]
    state['result'][state['fields'][state['curr']]] = msg
    state['curr'] = state['curr']+1
    if(state['curr'] == state['max']):
        endResponse = endFlow()
        return endResponse
    else:
        return state['fields'][state['curr']]


def endFlow():
    if flowStack.getSize() > 0:
        flow = flowStack.pop()
        endResponse = flowState[flow]['endMsg'] + json.dumps(flowState[flow]['result'])
        flowState[flow] = {}
        return endResponse

def cancelFlow():
    if flowStack.getSize() > 0:
        flow = flowStack.pop()
        return "Cancelled " + flow
    return "Didn't get you..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

