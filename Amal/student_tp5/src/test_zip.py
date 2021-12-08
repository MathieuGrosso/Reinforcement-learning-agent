# languages = ['Java', 'Python', 'JavaScript']
# versions = [14, 3, 6]

# result = dict(zip(languages, versions))
# print((result))
import torch
x=torch.tensor([1, 2, 1, 0, 1])
mask=torch.nonzero(x)
x=x[mask]
print(x)
