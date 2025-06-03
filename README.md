## CNN Inference with MPI Parallelization and Pipelining

This project implements Convolutional Neural Network (CNN) inference on the MNIST dataset using C and MPI (Message Passing Interface) for distributed execution across multiple CPUs.

## Parallel Inference Approaches

We implement and compare two approaches for distributed inference:

- **Data Parallelism** (`mnist_data.c`):  
  Each MPI process runs a full copy of the CNN model and processes a subset of the input data independently. This method is efficient and easy to scale with minimal communication overhead.

- **Pipeline Parallelism** (`mnist.c`):  
  Different CNN layers are distributed across MPI processes. Each process computes one or more layers and passes intermediate results to the next, forming a computation pipeline. This approach reduces idle time across layers and improves hardware utilization.

## Performance Comparison

While the **data parallel** method typically offers better performance due to lower communication overhead, the **pipeline parallel** method provides a novel execution strategy that enables simultaneous computation across layers. Despite the extra inter-process communication, our results show that the pipeline method achieves **comparable performance**, making it a viable alternative for specific system configurations.

In the following we show how to run the pipeline method.

### Usage

#### Train and Inference the simple network

First download the MNIST dataset files using the Makefile:

```sh
$ make get_mnist
```

compile the sources:

```sh
$ gcc cnn.c train_test.c -o train_test -lm
```
Then run training:

```sh
$ make train_mnist
```

If you want to get the results on testset:

```sh
$ make test_mnist
```

#### Inference of parallelised network

Now that you have the network trained, you can go for MPI program.
Compile the code by MPI compiler using:

```sh
$ /opt/mpich2/gnu/bin/mpicc -o mnist mnist.c cnn.c -std=c99 -lm
```

- In our case, the program was compiled and executed on a remote server, you should locate mpicc based on your system directory.

Run the code:

```sh
$ /opt/mpich2/gnu/bin/mpiexec.hydra -np <num_of_proc> ./mnist
```

### Description

The code comprises the following components:

- **CNN Model**: Implements a simple CNN model with layers for input, convolution, fully connected, and output.
- **MNIST Data Reading**: Parses the MNIST data files (images and labels) for both training and testing.
- **Training**: Trains the CNN model using the provided training data.
- **Testing**: Evaluates the trained model's performance using the provided test data.

### Dependencies
- Standard C libraries: `stdio.h`, `stdlib.h`, `stdint.h`, `assert.h`, `string.h`
- Additional helper functions: `endian.h`, `cnn.h`


### Important Notes
- Ensure that MPI is properly configured and accessible in your environment.
- If you face error:"'for' loop initial declaration used outside C99 mode", you can either declare the i outside the loop or use '-std=c99' for compilation.

### Acknowledgments
This program was developed using github.com/euske/nn1 repository. I extend my gratitude for their valuable contribution to the open-source community.

### Contributors
- [Reza Jahadi](https://github.com/rezajahadi)
- [Aye Sandar Thwe](https://github.com/ayesandarthwe)

