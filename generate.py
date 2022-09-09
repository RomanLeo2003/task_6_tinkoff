import pickle
import argparse
from train import Model

parser = argparse.ArgumentParser(description='Generate.py')
parser.add_argument('--model', type=str, help='Dir for model pickle file')
parser.add_argument('--prefix', type=str, help='Start-word')
parser.add_argument('--length', type=int, help='Len of generate text')
args = parser.parse_args()


with open(args.model, 'rb') as file:
    model = pickle.load(file)

print(model.generate(temp=0.3, prediction_len=args.length, start_text=args.prefix))
