## CNN Inference with MPI Parallelization

This C code implements Convolutional Neural Network (CNN) inference on the MNIST dataset. It utilizes Message Passing Interface (MPI) for parallelizing the inference process across multiple CPUs. The code is designed to read MNIST data files and perform training and testing of the CNN model.

### Usage
```sh
$ ./mnist train-images train-labels test-images test-labels
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

### How to Use
1. Compile the code using a C compiler with MPI support.
2. Run the compiled executable with appropriate arguments specifying paths to the MNIST data files.
3. Monitor training progress and evaluate testing accuracy.

### Important Notes
- Ensure that MPI is properly configured and accessible in your environment.
- MNIST data files should follow the IDX file format.

### Contributors
- This code was developed by [Author Name].

### License
This project is licensed under the [License Name] License - see the [LICENSE.md](LICENSE.md) file for details.

