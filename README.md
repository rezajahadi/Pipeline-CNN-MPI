## CNN Inference with MPI Parallelization and Pipelining

This C code implements Convolutional Neural Network (CNN) inference on the MNIST dataset. It utilizes Message Passing Interface (MPI) for parallelizing the inference process across multiple CPUs. Additionally, we have implemented pipelining to optimize the computation process.

In the traditional mode, each layer of the CNN would wait for the preceding layers to complete computation before proceeding. However, in our implementation, we assign layers to different CPUs. If the number of CPUs is less than the number of layers, the layers are divided among the CPUs. Conversely, if there are more CPUs than layers, additional CPUs are assigned to layers, resulting in data parallelism. This ensures that no layer remains idle during computation, and all layers contribute to the inference process simultaneously.

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

### Contributors
- [Reza Jahadi](https://github.com/rezajahadi)
- [Aye Sandar Thwe](https://github.com/ayesandarthwe)

