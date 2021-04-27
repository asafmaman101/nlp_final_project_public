import torch
from paraphrasing.utils import calculate_accuracy_and_f1

p = torch.load('/home/fodl/asafmaman/PycharmProjects/nlp_final_project_private/paraphrasing/evalutaion_results/positive_losses_quora_only.pt')
n = torch.load('/home/fodl/asafmaman/PycharmProjects/nlp_final_project_private/paraphrasing/evalutaion_results/negative_losses_quora_only.pt')

for thd in torch.linspace(1, 3, 10):
    calculate_accuracy_and_f1(n, p, thd)