import torch
model = torch._inductor.aoti_load_package('test.pt2')

# comment this out and then the second block will error
# I would expect this to error with both the first and second blocks
t = torch.rand(1, 36, 1024, 1024).to('cuda')
print("Input shape", t[:,0:9,...].shape)
model(t[:,0:9,...])


t = torch.rand(1, 9, 1024, 1024).to('cuda')
print("Input shape", t.shape)
model(t)