from keras.datasets import mnist, fashion_mnist, cifar10

permuted = True
augmented = True
show = True

# DATASET = "mnist"
# DATASET = "fashion"
DATASET = "cifar"
v = 1
vault = False
single_batch_mode = False
grid_shape = (2, 2)

input_shape = (32, 32, 3) if DATASET == 'cifar' else (28, 28, 1)


model_name = f"{'model_vault/' if vault else ''}{DATASET}-{'perm-' if permuted else ''}{'aug-' if augmented else ''}{grid_shape[0]}x{grid_shape[1]}-v{v}"
# random_states = get_states(model_name, grid_shape)
seed = 5555

info_dir = "training_info"

fashion_names = """
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
"""

cifar_names = """airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck"""
if DATASET == 'mnist':
    dataset = mnist
    classes = [i for i in range(10)]
elif DATASET == 'fashion':
    dataset = fashion_mnist
    classes = [c for c in fashion_names.split("\n") if c]
else:
    dataset = cifar10
    classes = [c for c in cifar_names.split("\n") if c]
