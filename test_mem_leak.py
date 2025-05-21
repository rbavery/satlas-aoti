import torch
model_path = "test.pt2"
model = torch._inductor.aoti_load_package(model_path)
device = torch.device("cuda" + ":" + str(torch.cuda.current_device()))
torch.cuda.set_device(device)
test_arr = torch.randn((16, 9 * 4, 1024, 1024), device=device)
for i in range(10):
    model(test_arr)
    print(i)
print("no leak")