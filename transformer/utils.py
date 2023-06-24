

def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    print("number of parameters: %.2fM" % (n_params / 1e6,))