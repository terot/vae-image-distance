import os.path
import observations

def input_dir(args):
  return  os.path.abspath(os.path.join(args.data_dir, "input",  args.dataset))

def model_checkpoint(args):
  param_path = "dim_latent-{}".format(args.dim_latent)
  return  os.path.abspath(os.path.join(args.data_dir, "models",  args.dataset, param_path))

def sample_dir(args):
  param_path = "dim_latent-{}".format(args.dim_latent)
  return  os.path.abspath(os.path.join(args.data_dir, "samples", args.dataset, param_path))

def output_dir(args):
  param_path = "dim_latent-{}".format(args.dim_latent)
  return  os.path.abspath(os.path.join(args.data_dir, "output", args.dataset, param_path))

def load_data(dataset, data_dir):  
  x_train = None
  x_test = None
  x_train_generator = None
  image_width = None
  image_height = None
  if dataset == "mnist":
    (x_train, _), (x_test, _) = observations.mnist(data_dir)
    return {
        "image_height": 28,
        "image_width": 28,
        "x_train": x_train,
        "x_test": x_test
    }    
  elif dataset == "fashion-mnist":
    (x_train, _), (x_test, _) = observations.fashion_mnist(data_dir)
    return {
        "image_height": 28,
        "image_width": 28,
        "x_train": x_train,
        "x_test": x_test
      }
  else:
    raise Exception("dataset not supported: {}".format(dataset))
