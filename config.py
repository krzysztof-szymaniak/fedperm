from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

seed = 0
split_seed = 0
batch_size = 128
epochs = 1000
EARLY_STOPPING_PATIENCE = 16
LR = 1e-3
augmented = True
sequentially = True
show = True
shape = 'overlap_cross_2x2'

# DATASET = "mnist"
# DATASET = "fashion"
DATASET = "cifar"
v = 1
vault = False
single_batch_mode = False

grid_shape = (2, 2)

input_shape = (32, 32, 3) if DATASET == 'cifar' else (28, 28, 1)
subinput_shape = (input_shape[0] // grid_shape[0], input_shape[1] // grid_shape[1], input_shape[2])


base_name = f"{'model_vault/' if vault else 'models/'}{DATASET}-{grid_shape[0]}x{grid_shape[1]}"
base_name += f"{'-perm' if seed != 0 else ''}"
base_name += f"{'-seq' if sequentially else ''}"
base_name += f"-{shape}"

model_name = f"{base_name}-v{v}"

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

# tensorboard --logdir
