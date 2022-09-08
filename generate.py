import pickle
import train
import argparse
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('--model', type=str, help='Dir for model pickle file')
parser.add_argument('--prefix', type=str, help='Start-word')
parser.add_argument('--length', type=int, help='Input dir for saving model')
parser.add_argument()
args = parser.parse_args()

seq, char_to_idx, idx_to_char = train.vectorizer.text_to_seq()

model_pickle = r"C:\Users\Роман\OneDrive\Рабочий стол\pickle_model.pkl"

with open(args.model, 'rb') as file:
    model = pickle.load(file)

print(model.generate(char_to_idx, idx_to_char, temp=0.3, prediction_len=args.length, start_text=args.prefix))
