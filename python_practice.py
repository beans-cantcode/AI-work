import torch
print("PyTorch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

data = [[1, 2], [3, 4]]

tensor_data = torch.tensor(data)

shape = (2,3)

randtens = torch.randn(10, requires_grad=True)

print(f"rand ={randtens}")

print(f"Tensor requires grad: {randtens.requires_grad}")

a = torch.tensor([[1.,2.,3.],[6.,5.,4.]])

b = torch.tensor([[7.,8.],[9.,10.],[11.,12.]])

c = a @ b

print(a)
print(b)
print(c)

snew_a = a.mean(dim=0)
dnew_a = a.mean(dim=1)

print(snew_a)
print(dnew_a)

nine_and_ten = b[1,:]

print(nine_and_ten)