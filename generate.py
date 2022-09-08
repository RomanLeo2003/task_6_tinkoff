import pickle
import train
import argparse
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('--model', type=str, help='Dir for model pickle file')
parser.add_argument('--prefix', type=str, help='Start-word')
parser.add_argument('--length', type=int, help='Input dir for saving model')
parser.add_argument()
args = parser.parse_args()

'''
def generate(pkl_filename, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text
'''


seq, char_to_idx, idx_to_char = train.vectorizer.text_to_seq()

model_pickle = r"C:\Users\Роман\OneDrive\Рабочий стол\pickle_model.pkl"

with open(args.model, 'rb') as file:
    model = pickle.load(file)

print(model.generate(char_to_idx, idx_to_char, temp=0.3, prediction_len=args.length, start_text=args.prefix))