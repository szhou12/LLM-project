from torch.nn.utils import prune
import copy
from transformers import AutoModelForCausalLM

SPARSITY_LIST = [.5, .9]

def get_weight_parameters(layer):
    '''
    Get all parameters/modules identified as 'weight'
    '''
    weight_parameters = []
    if len(list(layer.children())) > 0:
        for child in layer.children():
            for param in child.named_parameters():
                if 'weight' == param[0]:
                    # print(param)
                    weight_parameters.append((child, param[0]))
            weight_parameters.extend(get_weight_parameters(child))
    
    
    return weight_parameters


def prune_weight_parameters(model, prune_amount):
    '''
    Global pruning
    '''
    params_to_prune = get_weight_parameters(model)

    prune.global_unstructured(
        params_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=prune_amount,
    )

    for module, name in params_to_prune:
        try:
            prune.remove(module, name)
        except Exception as e:
            print(e)
    return model


def save_distilgpt2():
    '''
    Lightweight GPT2: 82M params
    '''
    name = 'distilgpt2'
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # save original model
    # model.save_pretrained(f"models/{name}")

    # save pruned models
    for sparsity in SPARSITY_LIST:
        model_to_prune = copy.deepcopy(model)
        pruned_model = prune_weight_parameters(model_to_prune, sparsity)
        pruned_model.save_pretrained(f"models/{name}_{sparsity}")


def save_gpt2():
    '''
    Meidum GPT2: 117M params
    '''
    name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # save original model
    # model.save_pretrained(f"models/{name}")

    # save pruned models
    for sparsity in SPARSITY_LIST:
        model_to_prune = copy.deepcopy(model)
        pruned_model = prune_weight_parameters(model_to_prune, sparsity)
        pruned_model.save_pretrained(f"models/{name}_{sparsity}")

def save_gpt_neo():
    '''
    Large GPT-3: 1.3B params
    '''
    name = 'gpt-neo-1.3B'
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # save original model
    # model.save_pretrained(f"models/{name}")

    # save pruned models
    for sparsity in SPARSITY_LIST:
        model_to_prune = copy.deepcopy(model)
        pruned_model = prune_weight_parameters(model_to_prune, sparsity)
        pruned_model.save_pretrained(f"models/{name}_{sparsity}")


if __name__ == "__main__":
    save_distilgpt2()
    save_gpt2()
    save_gpt_neo()
