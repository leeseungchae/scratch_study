import torch

from nlp.model.rnn import RNN

torch_input = torch.randn(
    3, 5, 10
)  # batch_first = False  sequence length , # batch size # hidden size
# print(torch_input)
# exit()
# torch_input = torch.randn(10,20,10) # batch_first = True  sbatch , sequence length  # hidden size
# batch_first = False(default)
rnn_custom = RNN(input_size=2, hidden_size=3, num_layers=2, bidirectional=False)

output,hn = rnn_custom(torch_input)

print(output)